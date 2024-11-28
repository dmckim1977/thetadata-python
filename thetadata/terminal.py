import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import httpx
import psutil
import wget

jdk_path = Path.home().joinpath('ThetaData').joinpath('ThetaTerminal') \
    .joinpath('jdk-19.0.1').joinpath('bin')

to_extract = Path.home().joinpath('ThetaData').joinpath('ThetaTerminal')

_thetadata_jar = "ThetaTerminal.jar"


class TerminalProcess:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.ready_event = Event()
        self.fpss_ready = Event()
        self.mdds_ready = Event()
        self.startup_failed = False
        self._monitor_thread = None
        self.startup_error = None

    def _monitor_output(self):
        """Monitor process output for startup markers."""
        try:
            while True:
                if not self.process or self.process.poll() is not None:
                    exit_code = self.process.poll() if self.process else None
                    self.startup_failed = True
                    self.startup_error = f"Process terminated with exit code: {exit_code}"
                    logging.error(
                        f"Terminal process terminated unexpectedly. Exit code: {exit_code}")
                    # Capture and log the process output for debugging
                    if self.process and self.process.stdout:
                        output = self.process.stdout.read()
                        logging.error(f"Process output: {output}")
                    break

                line = self.process.stdout.readline().strip()
                if not line:
                    continue

                logging.debug(f"Terminal output: {line}")

                if "FPSS] CONNECTED" in line:
                    logging.info("FPSS Connected")
                    self.fpss_ready.set()
                elif "MDDS] CONNECTED" in line or "[MDDS] Ready" in line:
                    logging.info("MDDS Connected")
                    self.mdds_ready.set()

                if self.fpss_ready.is_set() and self.mdds_ready.is_set():
                    self.ready_event.set()

        except Exception as e:
            self.startup_failed = True
            self.startup_error = str(e)
            logging.error(f"Error in monitor thread: {e}")
            raise

    def verify_services_ready(self) -> bool:
        """Verify both FPSS and MDDS services are actually ready to handle requests."""
        try:
            if not (self.fpss_ready.is_set() and self.mdds_ready.is_set()):
                logging.info("Waiting for services to be marked as ready...")
                return False

            with httpx.Client(timeout=5.0) as client:
                # Check basic connectivity
                mdds_response = client.get(
                    "http://127.0.0.1:25510/v2/system/mdds/status")
                mdds_response.raise_for_status()

                fpss_response = client.get(
                    "http://127.0.0.1:25510/v2/system/fpss/status")
                fpss_response.raise_for_status()

                # Test actual service readiness with a simple request
                test_response = client.get(
                    "http://127.0.0.1:25510/v2/list/roots/stock")
                test_response.raise_for_status()

                # Add a small delay after successful verification
                time.sleep(1.0)  # 1 second delay after verification

                logging.info("Services verified and ready for requests")
                return True

        except Exception as e:
            logging.error(f"Service verification failed: {e}")
            return False

    def terminate(self):
        """Terminate the terminal process and all its children."""
        if self.process:
            try:
                # On Windows, we need to kill child processes first
                if platform.system() == 'Windows':
                    process = psutil.Process(self.process.pid)
                    for child in process.children(recursive=True):
                        try:
                            child.terminate()
                            # Give it a moment to terminate gracefully
                            child.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass

                # Terminate the main process
                self.process.terminate()
                try:
                    # Wait for process to terminate
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    self.process.kill()
                    self.process.wait()

            except Exception as e:
                logging.error(f"Error terminating process: {e}")
            finally:
                self.process = None

    def start(self, cwd: Path, username: str, passwd: str,
              jvm_mem: int = 0) -> bool:
        """Start the terminal process and wait for successful initialization."""
        cmd = ["java"]
        if jvm_mem > 0:
            cmd.extend([f"-Xmx{jvm_mem}G"])
        cmd.extend(["-jar", "ThetaTerminal.jar", username, passwd])

        logging.info(f"Starting terminal from directory: {cwd}")
        logging.info(f"Command: {' '.join(cmd)}")

        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == 'Windows' else 0
            )

            # Start monitoring thread
            self._monitor_thread = Thread(target=self._monitor_output,
                                          daemon=True)
            self._monitor_thread.start()

            # Wait for service markers
            if not self.ready_event.wait(timeout=60):
                logging.error("Timeout waiting for services to start")
                self.terminate()
                return False

            # Verify services are actually ready
            if not self.verify_services_ready():
                logging.error("Services failed readiness verification")
                self.terminate()
                return False

            logging.info("Terminal startup successful")
            return True

        except Exception as e:
            logging.error(f"Failed to start terminal process: {e}")
            self.terminate()
            return False

def bar_progress(current, total, width=80):
    progress_message = "Downloading open-jdk 19.0.1  -->  %d%% Complete" % (current / total * 100)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def _install_jdk() -> bool:
    url_windows = 'https://download.java.net/java/GA/jdk19.0.1/afdd2e245b014143b62ccb916125e3ce/10/GPL/openjdk-19.0.1_windows-x64_bin.zip'
    if jdk_path.exists():
        return True
    try:
        if platform.system() == 'Windows':
            print('--------------------------------------------------------------\n')
            print('Initiated first time setup, do not terminate the program!')
            print('\n--------------------------------------------------------------')
            download = wget.download(url_windows, bar=bar_progress)

            with zipfile.ZipFile(download, 'r') as zip_ref:
                zip_ref.extractall(to_extract)
            os.remove(download)
            print()
            return True
    except:
        pass
    return False


def _verify_java():
    if not shutil.which("java"):
        print('Java 11 or higher is required to use this API. Please install Java on this machine.')
        exit(1)
    # version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
    # pattern = r'\"(\d+\.\d+).*\"'
    # version = float(re.search(pattern, version.decode('utf8')).groups()[0])

    # if version < 11:
    #    print('Java 11 or higher is required to use this API. You are using Java '
    #          + str(version) + '. Please upgrade to a newer version.')
    #    exit(1)


def launch_terminal(username: str = None, passwd: str = None, use_bundle: bool = True, jvm_mem: int = 0, move_jar: bool = True):
    cwd = None
    use_it = False

    if use_bundle:
        use_it = _install_jdk()

    if use_it:
        cwd = jdk_path
        if move_jar:
            shutil.move("ThetaTerminal.jar", str(cwd.joinpath('ThetaTerminal.jar')))
    # else:
    #    _verify_java()

    if jvm_mem > 0:
        if os.name != 'nt':
            process = subprocess.Popen([f"java -Xmx{jvm_mem}G -jar ThetaTerminal.jar {username} {passwd}"],
                                       stdout=subprocess.PIPE, shell=True)
        else:
            process = subprocess.Popen(["java", f"-Xmx{jvm_mem}G", "-jar", "ThetaTerminal.jar", username, passwd],
                                       stdout=subprocess.PIPE, shell=True, cwd=cwd)
    else:
        if os.name != 'nt':
            process = subprocess.Popen([f"java -jar ThetaTerminal.jar {username} {passwd}"],
                                       stdout=subprocess.PIPE, shell=True)
        else:
            process = subprocess.Popen(["java", "-jar", "ThetaTerminal.jar", username, passwd],
                                       stdout=subprocess.PIPE, shell=True, cwd=cwd)
    for line in process.stdout:
        print(line.decode('utf-8').rstrip("\n"))


def check_download(auto_update: bool, stable: bool) -> bool:
    if stable:
        link = 'http://download-stable.thetadata.us'
    else:
        link = 'http://download-unstable.thetadata.us'
    try:
        if not os.path.exists('ThetaTerminal.jar') or auto_update:
            jar = urllib.request.urlopen(link)
            with open('ThetaTerminal.jar', 'wb') as output:
                output.write(jar.read())
                output.close()
        return True
    except:
        try:
            if not os.path.exists('ThetaTerminal.jar') or auto_update:
                jar = urllib.request.urlopen(link)
                with open('ThetaTerminal.jar', 'wb') as output:
                    output.write(jar.read())
                    output.close()
        except:
            print('Unable to fetch the latest terminal version. '
                  'Please contact support.')
    return False


def kill_existing_terminal() -> None:
    """Utility function to kill any ThetaData terminal processes by
    scanning all running proceeses and killing such process

    """
    for pid in psutil.pids():
        try:
            cmdline_args = psutil.Process(pid=pid).cmdline()
            for arg in cmdline_args:
                if _thetadata_jar in arg:
                    os.kill(pid, signal.SIGTERM)
        except:
            pass


def is_terminal_instance_running() -> bool:
    """
    Checks if thetadata terminal is running or not
    Returns:
        bool: True if running else False
    """
    running = False
    for pid in psutil.pids():
        try:
            cmdline_args = psutil.Process(pid=pid).cmdline()
            for arg in cmdline_args:
                if _thetadata_jar in arg:
                    running = True
                    break
        except:
            pass
    return running

