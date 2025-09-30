import base64
import mimetypes
import webview
import threading
import json
import requests
import os
import io
import time
import platform
import subprocess
import re
import zipfile
import tarfile
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pyaudio
import numpy as np



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tts_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalServerManager:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        self.server_process = None
        self.setup_cancelled = False
        
        # Configuration
        self.github_repo = "your-org/tts-server"  # Update this to actual repo
        self.release_tag = "latest"
        self.server_port = 8000
        self.os_name, self.arch = self._get_os_arch()
        self.install_path = self._get_applications_folder(self.os_name)
        self.executable_name = "tts-server.exe" if self.os_name == "windows" else "tts-server"
        self.executable_path = os.path.join(self.install_path, self.executable_name)
        self.config_path = os.path.join(self.install_path, "config.json")
        self.log_path = os.path.join(self.install_path, "server.log")
        
    def cancel_setup(self):
        """Cancel ongoing setup."""
        self.setup_cancelled = True
        logger.info("Setup cancellation requested")

    def _get_os_arch(self):
        """Enhanced OS/arch detection."""
        os_name = platform.system().lower()
        arch = platform.machine().lower()

        # Normalize OS names
        if os_name == "darwin": 
            os_name = "macos"
        elif os_name not in ["windows", "linux"]:
            raise ValueError(f"Unsupported OS: {os_name}")

        # Normalize architecture
        if arch in ["x86_64", "amd64"]: 
            arch = "x64"
        elif arch in ["arm64", "aarch64"]: 
            arch = "arm64"
        elif arch in ["i386", "i686"]:
            arch = "x86"
        else:
            logger.warning(f"Unknown architecture {arch}, assuming x64")
            arch = "x64"
        
        return os_name, arch

    def _get_applications_folder(self, os_name):
        """Get installation directory with proper permissions."""
        if os_name == "windows":
            base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
            install_dir = os.path.join(base_dir, 'DeepFak3rTTSServer')
        elif os_name == "macos":
            install_dir = os.path.expanduser('~/Applications/DeepFak3rTTSServer')
        else:  # linux
            install_dir = os.path.expanduser('~/.local/share/deepfak3r-tts-server')
        
        # Ensure directory exists and is writable
        Path(install_dir).mkdir(parents=True, exist_ok=True)
        
        if not os.access(install_dir, os.W_OK):
            raise PermissionError(f"No write access to {install_dir}")
            
        return install_dir

    def _get_download_url(self, os_name, arch):
        """Get download URL with fallback options."""
        base_url = f"https://github.com/{self.github_repo}/releases/{self.release_tag}/download"
        filename = f"deepfak3rvoice-tts-server-{os_name}-{arch}"
        ext = "zip" if os_name == "windows" else "tar.gz"
        
        return f"{base_url}/{filename}.{ext}", f"{filename}.{ext}"

    def _verify_download(self, file_path, expected_checksum=None):
        """Verify downloaded file integrity."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Downloaded file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            raise ValueError(f"Downloaded file too small: {file_size} bytes")
            
        # TODO: Implement checksum verification if checksums are provided
        if expected_checksum:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != expected_checksum:
                    raise ValueError(f"Checksum mismatch for {file_path}")
        
        logger.info(f"Download verified: {file_path} ({file_size:,} bytes)")

    def setup_server(self):
        """Enhanced server setup with comprehensive error handling."""
        try:
            if self.setup_cancelled:
                return

            logger.info(f"Setting up server for {self.os_name}-{self.arch} in {self.install_path}")

            # Check if already installed and working
            if os.path.exists(self.executable_path):
                logger.info("Server executable found, testing...")
                if self._test_existing_installation(self.executable_path):
                    logger.info("Existing installation is working")
                    self.progress_callback({
                        "stage": "starting",
                        "progress": 90,
                        "message": "Starting existing server installation..."
                    })
                    self.start_server()
                    return
                else:
                    logger.warning("Existing installation not working, reinstalling...")

            # Download
            download_url, filename = self._get_download_url(self.os_name, self.arch)
            archive_path = os.path.join(self.install_path, filename)
            
            logger.info(f"Downloading from: {download_url}")
            self._download_with_progress(download_url, archive_path)
            
            if self.setup_cancelled:
                return

            # Verify download
            self._verify_download(archive_path)

            # Extract
            self._extract_with_progress(archive_path, self.install_path)
            
            if self.setup_cancelled:
                return

            # Cleanup
            try:
                os.remove(archive_path)
            except:
                pass

            # Make executable (Unix-like systems)
            if self.os_name != "windows":
                os.chmod(self.executable_path, 0o755)

            # Start server
            self.start_server()

        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            self.progress_callback({
                "stage": "error",
                "progress": 0,
                "message": "Setup failed",
                "error": str(e)
            })

    def check_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status information."""
        is_installed = os.path.exists(self.executable_path)
        is_running = self.server_process is not None and self.server_process.poll() is None
        
        # Get version if installed
        version = None
        if is_installed:
            try:
                result = subprocess.run(
                    [self.executable_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
            except Exception:
                version = "1.0.0"  # Default version
        
        # Get PID if running
        pid = None
        if is_running:
            pid = self.server_process.pid

        return {
            "isInstalled": is_installed,
            "version": version or "1.0.0",
            "installPath": self.install_path,
            "executablePath": self.executable_path,
            "configPath": self.config_path,
            "logPath": self.log_path,
            "pid": pid,
            "port": self.server_port,
            "isRunning": is_running,
            "lastStarted": time.time() if is_running else None
        }

    def _test_existing_installation(self, executable_path):
        """Test if existing installation works."""
        try:
            # Try to run with --version flag
            result = subprocess.run(
                [executable_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to test existing installation: {e}")
            return False

    def _download_with_progress(self, url, file_path):
        """Download with progress reporting and resume support."""
        self.progress_callback({
            "stage": "downloading",
            "progress": 0,
            "message": "Starting download..."
        })

        headers = {}
        resume_pos = 0
        
        # Check if partial file exists
        if os.path.exists(file_path):
            resume_pos = os.path.getsize(file_path)
            headers['Range'] = f'bytes={resume_pos}-'
            logger.info(f"Resuming download from byte {resume_pos}")

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Get total size
            content_length = response.headers.get('content-length')
            if content_length:
                total_size = int(content_length) + resume_pos
            else:
                total_size = 0

            mode = 'ab' if resume_pos > 0 else 'wb'
            
            with open(file_path, mode) as f:
                downloaded = resume_pos
                
                for chunk in response.iter_content(chunk_size=8192):
                    if self.setup_cancelled:
                        return
                        
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 80)  # Use 80% for download
                            self.progress_callback({
                                "stage": "downloading",
                                "progress": min(progress, 79),
                                "message": f"Downloading... {downloaded:,}/{total_size:,} bytes"
                            })

            logger.info(f"Download completed: {file_path}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            raise Exception(f"Download failed: {e}")

    def _extract_with_progress(self, archive_path, extract_path):
        """Extract archive with progress reporting."""
        self.progress_callback({
            "stage": "extracting",
            "progress": 80,
            "message": "Extracting files..."
        })

        try:
            if archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    members = zip_ref.namelist()
                    total_files = len(members)
                    
                    for i, member in enumerate(members):
                        if self.setup_cancelled:
                            return
                            
                        zip_ref.extract(member, extract_path)
                        progress = 80 + int((i + 1) / total_files * 10)  # 80-90%
                        self.progress_callback({
                            "stage": "extracting",
                            "progress": progress,
                            "message": f"Extracting... {i+1}/{total_files} files"
                        })
            
            else:  # tar.gz
                with tarfile.open(archive_path, "r:gz") as tar_ref:
                    members = tar_ref.getmembers()
                    total_files = len(members)
                    
                    for i, member in enumerate(members):
                        if self.setup_cancelled:
                            return
                            
                        tar_ref.extract(member, extract_path)
                        progress = 80 + int((i + 1) / total_files * 10)  # 80-90%
                        self.progress_callback({
                            "stage": "extracting",
                            "progress": progress,
                            "message": f"Extracting... {i+1}/{total_files} files"
                        })

            self.progress_callback({
                "stage": "extracting",
                "progress": 90,
                "message": "Extraction complete"
            })
            
            logger.info(f"Extraction completed: {extract_path}")
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise Exception(f"Extraction failed: {e}")

    def start_server(self):
        """Start server with proper process management."""
        if self.setup_cancelled:
            return
            
        self.progress_callback({
            "stage": "starting",
            "progress": 90,
            "message": "Starting TTS server..."
        })

        try:
            # Kill any existing server process
            self._kill_existing_server()

            logger.info(f"Starting server: {self.executable_path}")
            
            # Determine startup flags based on OS
            startup_flags = {}
            if platform.system() == "Windows":
                startup_flags['creationflags'] = subprocess.DETACHED_PROCESS
            else:
                startup_flags['close_fds'] = True
                
            # Start server process
            cmd = [self.executable_path, "--host", "0.0.0.0", "--port", str(self.server_port)]
            
            # Redirect output to log file
            log_file = open(self.log_path, 'w')
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                **startup_flags
            )

            logger.info(f"Server started with PID: {self.server_process.pid}")

            # Wait for server to be ready
            if self._wait_for_server_ready():
                self.progress_callback({
                    "stage": "complete",
                    "progress": 100,
                    "message": f"Server started successfully on port {self.server_port}"
                })
            else:
                raise Exception("Server failed to become ready")

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.progress_callback({
                "stage": "error",
                "progress": 0,
                "message": "Failed to start server",
                "error": str(e)
            })

    def stop_server(self):
        """Stop the running server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                self.server_process = None
                logger.info("Local server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process = None
                logger.info("Local server forcefully stopped")
            except Exception as e:
                logger.error(f"Failed to stop local server: {e}")

    def _kill_existing_server(self):
        """Kill any existing server processes on the target port."""
        try:
            if platform.system() == "Windows":
                # Windows: Find and kill process using port
                result = subprocess.run(
                    f'netstat -ano | findstr :{self.server_port}',
                    shell=True, capture_output=True, text=True
                )
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'LISTENING' in line:
                            pid = line.split()[-1]
                            try:
                                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                                logger.info(f"Killed existing process PID {pid}")
                            except:
                                pass
            else:
                # Unix-like: Kill process using port
                try:
                    result = subprocess.run(
                        ['lsof', '-ti', f'tcp:{self.server_port}'],
                        capture_output=True, text=True
                    )
                    if result.stdout:
                        pids = result.stdout.strip().split('\n')
                        for pid in pids:
                            if pid:
                                subprocess.run(['kill', '-9', pid])
                                logger.info(f"Killed existing process PID {pid}")
                except FileNotFoundError:
                    # lsof not available
                    pass
        except Exception as e:
            logger.warning(f"Failed to kill existing server: {e}")

    def _wait_for_server_ready(self, timeout=60):
        """Wait for server to be ready with health checks."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.setup_cancelled:
                return False
                
            try:
                # Check if process is still running
                if self.server_process.poll() is not None:
                    logger.error("Server process terminated unexpectedly")
                    return False
                
                # Try health check
                response = requests.get(
                    f'http://localhost:{self.server_port}/health',
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("Server health check passed")
                    return True
                    
            except requests.exceptions.RequestException:
                # Server not ready yet, continue waiting
                pass
            
            elapsed = int(time.time() - start_time)
            self.progress_callback({
                "stage": "starting",
                "progress": min(90 + int((elapsed / timeout) * 10), 99),
                "message": f"Waiting for server to be ready... ({elapsed}s/{timeout}s)"
            })
            
            time.sleep(2)
        
        logger.error(f"Server failed to become ready within {timeout} seconds")
        return False

class VirtualAudioManager:
    """Manages virtual microphone detection and audio streaming"""
    
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.current_stream = None
        self.stop_streaming = False
        self.sample_rate = 24000
        self.channels = 1
        
    def get_virtual_microphones(self) -> List[Dict[str, Any]]:
        """Get list of available virtual microphones"""
        virtual_mics = []
        
        try:
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                device_name = device_info.get('name', '').lower()
                
                if device_info.get('maxOutputChannels', 0) > 0:
                    virtual_patterns = [
                        'virtual', 'vb-audio', 'voicemeeter', 'cable',
                        'obs', 'elgato', 'streamlabs', 'blackhole',
                        'soundflower', 'jack', 'asio', 'wasapi'
                    ]
                    
                    is_virtual = any(pattern in device_name for pattern in virtual_patterns)
                    
                    if is_virtual or 'virtual' in device_name or 'cable' in device_name:
                        virtual_mics.append({
                            'id': i,
                            'name': device_info.get('name', f'Device {i}'),
                            'channels': device_info.get('maxOutputChannels', 0),
                            'sample_rate': device_info.get('defaultSampleRate', 44100),
                            'device_index': i
                        })
            
            # Add common virtual mics if not detected
            common_virtuals = [
                {'id': 'vb-cable', 'name': 'VB-Audio Virtual Cable', 'channels': 2, 'sample_rate': 48000},
                {'id': 'voicemeeter', 'name': 'VoiceMeeter Input', 'channels': 2, 'sample_rate': 48000},
            ]
            
            for common in common_virtuals:
                if not any(common['name'].lower() in vm['name'].lower() for vm in virtual_mics):
                    virtual_mics.append(common)
                    
        except Exception as e:
            print(f"Error getting virtual microphones: {e}")
            
        return virtual_mics
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.p:
            self.p.terminate()

class TTSStreamingAPI:
    """API bridge for TTS streaming functionality"""
    
    def __init__(self):
        self.virtual_audio = VirtualAudioManager()
        self.streaming_active = False
        self.current_session_id = None
        
    def get_virtual_microphones(self) -> Dict[str, Any]:
        """Get available virtual microphones"""
        try:
            virtual_mics = self.virtual_audio.get_virtual_microphones()
            return {
                "success": True,
                "virtual_mics": virtual_mics
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        return {
            "success": True,
            "streaming_active": self.streaming_active,
            "has_stream": self.virtual_audio.current_stream is not None
        }
        
    def generate_tts(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Original TTS generation (non-streaming)"""
        try:
            text = request_data.get("text", "")
            voice_data = request_data.get("voice_blob_data", "")
            server_url = request_data.get("server_url", "")
            
            # Decode the voice data
            voice_bytes = base64.b64decode(voice_data)
            
            # Create files dict for requests
            files = {
                'voice_file': ('voice_sample.wav', io.BytesIO(voice_bytes), 'audio/wav')
            }
            data = {'text': text}
            
            # Make request to TTS server
            response = requests.post(f"{server_url}/single-speaker", files=files, data=data, timeout=30)
            
            if response.ok:
                audio_data = base64.b64encode(response.content).decode('utf-8')
                return {
                    "success": True,
                    "audio_data": audio_data,
                    "content_type": response.headers.get('content-type', 'audio/wav')
                }
            else:
                return {
                    "success": False,
                    "error": f"Server error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_virtual_mic_streaming()
        self.virtual_audio.cleanup()
        
class Api:
    def __init__(self):
        self._active_runpod_instance: Optional[Dict[str, str]] = None
        self.local_server_manager = LocalServerManager(self.send_progress_to_js)
        self.tts_api = TTSStreamingAPI()
        self.window = None
        self._setup_thread: Optional[threading.Thread] = None

    def set_window(self, window):
        self.window = window

    def send_progress_to_js(self, progress_data):
        """Enhanced progress reporting with error handling."""
        try:
            if self.window:
                js_code = f'window.dispatchLocalServerProgress({json.dumps(progress_data)})'
                self.window.evaluate_js(js_code)
                logger.info(f"Progress update sent: {progress_data['stage']} - {progress_data['progress']}%")
        except Exception as e:
            logger.error(f"Failed to send progress to JS: {e}")

    def start_local_server_setup(self):
        """Thread-safe server setup with better error handling."""
        try:
            if self._setup_thread and self._setup_thread.is_alive():
                logger.warning("Setup already in progress")
                return

            self._setup_thread = threading.Thread(
                target=self._safe_setup_wrapper,
                daemon=True
            )
            self._setup_thread.start()
            logger.info("Local server setup thread started")
        except Exception as e:
            logger.error(f"Failed to start setup thread: {e}")
            self.send_progress_to_js({
                "stage": "error",
                "progress": 0,
                "message": "Failed to start setup",
                "error": str(e)
            })

    def _safe_setup_wrapper(self):
        """Wrapper for setup with comprehensive error handling."""
        try:
            self.local_server_manager.setup_server()
        except Exception as e:
            logger.error(f"Setup failed with exception: {e}", exc_info=True)
            self.send_progress_to_js({
                "stage": "error",
                "progress": 0,
                "message": "Setup failed unexpectedly",
                "error": str(e)
            })

    def check_local_server(self):
        """
        Called by JS to get the current server status.
        This must return a dict matching the LocalServerInfo interface in TS.
        """
        try:
            info = self.local_server_manager.check_server_status()
            logger.info(f"Local server status: {info}")
            return info
        except Exception as e:
            logger.error(f"Error checking local server: {e}")
            # Return a default error state if checks fail
            return {
                "isInstalled": False,
                "isRunning": False,
                "port": 8000,
                "version": "unknown",
                "installPath": "",
                "executablePath": "",
                "configPath": "",
                "logPath": "",
                "pid": None,
                "lastStarted": None
            }

    def start_local_server(self):
        """Called by JS to start the installed local server."""
        try:
            if not self.local_server_manager.server_process or self.local_server_manager.server_process.poll() is not None:
                self.local_server_manager.start_server()
            else:
                logger.info("Server is already running")
        except Exception as e:
            logger.error(f"Failed to start local server: {e}")
            raise

    def stop_local_server(self):
        """Called by JS to stop the running local server."""
        try:
            self.local_server_manager.stop_server()
        except Exception as e:
            logger.error(f"Failed to stop local server: {e}")
            raise

    def set_active_runpod_instance(self, api_key: str, pod_id: str):
        """
        Called by JS to set the active RunPod instance.
        This allows the Python TTS generation logic to know where to send requests.
        """
        self._active_runpod_instance = {"apiKey": api_key, "podId": pod_id}
        logger.info(f"Active RunPod instance set: {pod_id}")

    def get_active_runpod_instance(self):
        """Get the currently active RunPod instance."""
        return self._active_runpod_instance
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect system hardware and return comprehensive information."""
        os_name, arch = self._get_os_arch()
        
        GPU_DATABASE = {
            # NVIDIA RTX 50 Series (2025)
            "rtx 5090": {"vramGB": 32, "performanceScore": 100},
            "rtx 5080": {"vramGB": 16, "performanceScore": 85},
            "rtx 5070 ti": {"vramGB": 16, "performanceScore": 79},
            "rtx 5070": {"vramGB": 12, "performanceScore": 68},
            "rtx 5060 ti": {"vramGB": 16, "performanceScore": 54},
            "rtx 5060": {"vramGB": 8, "performanceScore": 45},
            
            # NVIDIA RTX 40 Series
            "rtx 4090": {"vramGB": 24, "performanceScore": 95},
            "rtx 4080 super": {"vramGB": 16, "performanceScore": 83},
            "rtx 4080": {"vramGB": 16, "performanceScore": 82},
            "rtx 4070 ti super": {"vramGB": 16, "performanceScore": 74},
            "rtx 4070 ti": {"vramGB": 12, "performanceScore": 71},
            "rtx 4070 super": {"vramGB": 12, "performanceScore": 67},
            "rtx 4070": {"vramGB": 12, "performanceScore": 63},
            "rtx 4060 ti": {"vramGB": 16, "performanceScore": 55},
            "rtx 4060": {"vramGB": 8, "performanceScore": 50},
            
            # AMD RX 9000 Series (2025)
            "rx 9070 xt": {"vramGB": 16, "performanceScore": 80},
            "rx 9070": {"vramGB": 16, "performanceScore": 75},
            "rx 9060 xt": {"vramGB": 16, "performanceScore": 60},
            
            # AMD RX 7000/8000 Series
            "rx 7900 xtx": {"vramGB": 24, "performanceScore": 85},
            "rx 7900 xt": {"vramGB": 20, "performanceScore": 80},
            "rx 7800 xt": {"vramGB": 16, "performanceScore": 70},
            "rx 7700 xt": {"vramGB": 12, "performanceScore": 60},
            "rx 7600 xt": {"vramGB": 16, "performanceScore": 55},
            
            # Intel Arc Battlemage (2025)
            "arc b580": {"vramGB": 12, "performanceScore": 50},
            "arc b570": {"vramGB": 10, "performanceScore": 45},
            "arc b560": {"vramGB": 8, "performanceScore": 40},
            
            # Apple Silicon (for macOS)
            "apple m4 pro": {"vramGB": 24, "performanceScore": 70},
            "apple m4": {"vramGB": 16, "performanceScore": 60},
            "apple m3 pro": {"vramGB": 18, "performanceScore": 65},
            "apple m3": {"vramGB": 8, "performanceScore": 55},
            "apple m2 ultra": {"vramGB": 64, "performanceScore": 80},
            "apple m2 pro": {"vramGB": 32, "performanceScore": 70},
            "apple m2": {"vramGB": 8, "performanceScore": 50},
            "apple m1 ultra": {"vramGB": 64, "performanceScore": 75},
            "apple m1 pro": {"vramGB": 32, "performanceScore": 65},
            "apple m1": {"vramGB": 8, "performanceScore": 50},
        }
        
        gpus = []
        
        try:
            if os_name == "windows" or os_name == "linux":
                # NVIDIA
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if ',' in line:
                                name, vram_str = line.split(',', 1)
                                model = name.lower().strip().replace('nvidia geforce ', '').replace('geforce ', '')
                                vram_gb = int(vram_str.strip().split()[0]) / 1024
                                specs = GPU_DATABASE.get(model, {"vramGB": vram_gb, "performanceScore": 50})
                                gpus.append({
                                    "vendor": "nvidia",
                                    "model": model,
                                    "vramGB": specs["vramGB"],
                                    "isSupported": specs["vramGB"] >= 8,
                                    "performanceScore": specs["performanceScore"]
                                })
                except Exception:
                    pass
                
                # AMD (rocm-smi if installed)
                try:
                    result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo', 'vram'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        lines = result.stdout.lower().split('\n')
                        for line in lines:
                            if 'product name' in line and ':' in line:
                                model = line.split(':', 1)[1].strip().replace('radeon rx ', 'rx ')
                                vram_gb = 8  # Default fallback
                                for vram_line in lines:
                                    if 'vram total' in vram_line and 'mb' in vram_line:
                                        try:
                                            vram_mb = int(re.search(r'(\d+)', vram_line).group(1))
                                            vram_gb = vram_mb / 1024
                                        except:
                                            pass
                                specs = GPU_DATABASE.get(model, {"vramGB": vram_gb, "performanceScore": 50})
                                gpus.append({
                                    "vendor": "amd",
                                    "model": model,
                                    "vramGB": specs["vramGB"],
                                    "isSupported": specs["vramGB"] >= 8,
                                    "performanceScore": specs["performanceScore"]
                                })
                                break
                except Exception:
                    pass
                
                # Intel Arc
                try:
                    result = subprocess.run(['clinfo'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and 'intel' in result.stdout.lower():
                        model_match = re.search(r'intel.*arc\s+([^\n]+)', result.stdout.lower())
                        if model_match:
                            model = f"arc {model_match.group(1).strip()}"
                            vram_gb = 8  # Default for Arc cards
                            specs = GPU_DATABASE.get(model, {"vramGB": vram_gb, "performanceScore": 40})
                            gpus.append({
                                "vendor": "intel",
                                "model": model,
                                "vramGB": specs["vramGB"],
                                "isSupported": specs["vramGB"] >= 8,
                                "performanceScore": specs["performanceScore"]
                            })
                except Exception:
                    pass
            
            elif os_name == "macos":
                # Apple Silicon
                try:
                    result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                          capture_output=True, text=True, timeout=15)
                    if result.returncode == 0:
                        lines = result.stdout.lower().split('\n')
                        for line in lines:
                            if 'chipset model' in line and ':' in line:
                                model = line.split(':', 1)[1].strip()
                                if 'apple' in model:
                                    # Extract memory info
                                    vram_gb = 8  # Default
                                    for mem_line in lines:
                                        if 'vram' in mem_line and 'gb' in mem_line:
                                            vram_match = re.search(r'(\d+)\s*gb', mem_line)
                                            if vram_match:
                                                vram_gb = int(vram_match.group(1))
                                                break
                                    
                                    specs = GPU_DATABASE.get(model, {"vramGB": vram_gb, "performanceScore": 60})
                                    gpus.append({
                                        "vendor": "apple",
                                        "model": model,
                                        "vramGB": specs["vramGB"],
                                        "isSupported": specs["vramGB"] >= 8,
                                        "performanceScore": specs["performanceScore"]
                                    })
                                    break
                except Exception:
                    pass
        
        except Exception as e:
            logger.error(f"Hardware detection error: {e}")
        
        # Calculate totals
        total_vram = sum(gpu["vramGB"] for gpu in gpus)
        has_gpu = len(gpus) > 0
        
        # Determine recommended setup
        if has_gpu and total_vram >= 12:
            recommended_setup = "local"
        elif has_gpu and total_vram >= 6:
            recommended_setup = "remote"
        else:
            recommended_setup = "runpod"
        
        return {
            "os": os_name,
            "arch": arch,
            "hasGPU": has_gpu,
            "gpus": gpus,
            "totalVRAM": total_vram,
            "recommendedSetup": recommended_setup
        }
    
    def _get_os_arch(self):
        """Get normalized OS and architecture."""
        os_name = platform.system().lower()
        arch = platform.machine().lower()

        if os_name == "darwin": 
            os_name = "macos"
        elif os_name not in ["windows", "linux"]:
            os_name = "linux"  # Default fallback

        if arch in ["x86_64", "amd64"]: 
            arch = "x64"
        elif arch in ["arm64", "aarch64"]: 
            arch = "arm64"
        elif arch in ["i386", "i686"]:
            arch = "x86"
        else:
            arch = "x64"  # Default fallback
        
        return os_name, arch
    
    def select_audio_file(self) -> Dict[str, Any]:
        """
        Open a file dialog to select an audio file using PyWebView's built-in dialog.
        """
        try:
            import webview
            
            # Use PyWebView's built-in file dialog
            file_types = (
                'Audio Files (*.mp3;*.wav;*.m4a;*.aac;*.ogg)',
                'All files (*.*)',
            )
            
            result = webview.windows[0].create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=file_types
            )
            
            if not result or len(result) == 0:
                return {
                    "success": False,
                    "error": "No file selected"
                }
            
            file_path = result[0]
            
            # Read and encode file
            with open(file_path, 'rb') as file:
                file_data = base64.b64encode(file.read()).decode('utf-8')
            
            # Get file info
            filename = os.path.basename(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            
            if mime_type is None:
                mime_type = "audio/mpeg"
            
            return {
                "success": True,
                "filename": filename,
                "file_data": file_data,
                "mime_type": mime_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def generate_tts(self, request_data):
        """Original TTS generation (non-streaming)"""
        return self.tts_api.generate_tts(request_data)

    def get_virtual_microphones(self):
        """Get available virtual microphones"""
        return self.tts_api.get_virtual_microphones()
    
    def get_streaming_status(self):
        """
        Get current streaming status
        
        Returns:
            dict: {
                "success": bool,
                "streaming_active": bool,
                "has_stream": bool
            }
        """
        return self.tts_api.get_streaming_status()
    
    def cleanup_on_exit(self):
        """
        Cleanup when app is closing
        Called automatically on window close
        """
        print("🧹 Cleaning up TTS API resources...")
        self.tts_api.cleanup()
        return {"success": True, "message": "Cleanup completed"}

def on_closing():
    """Enhanced application exit handler."""
    logger.info("Application is closing...")
    
    try:
        # Terminate RunPod instances to save costs
        if hasattr(api, '_active_runpod_instance') and api._active_runpod_instance:
            logger.info(f"Terminating RunPod instance {api._active_runpod_instance['podId']} to save costs...")
            try:
                terminate_runpod_instance(
                    api._active_runpod_instance['apiKey'], 
                    api._active_runpod_instance['podId']
                )
                logger.info("RunPod instance termination request sent")
            except Exception as e:
                logger.error(f"Failed to terminate RunPod instance: {e}")
        
        # Stop local server if running
        if hasattr(api, 'local_server_manager') and api.local_server_manager:
            api.local_server_manager.stop_server()
                
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")
    
    logger.info("Application shutdown complete")

def terminate_runpod_instance(api_key, pod_id):
    """Terminate RunPod instance via GraphQL API."""
    url = "https://api.runpod.ai/graphql"
    
    query = """
    mutation terminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input) {
            id
        }
    }
    """
    
    variables = {
        "input": {"podId": pod_id}
    }
    
    response = requests.post(
        url,
        json={"query": query, "variables": variables},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        timeout=30
    )
    
    response.raise_for_status()
    
    data = response.json()
    if data.get("errors"):
        raise Exception(f"GraphQL error: {data['errors'][0]['message']}")
    
    return data["data"]["podTerminate"]

if __name__ == '__main__':
    try:
        api = Api()
        
        # Create window with proper configuration
        window = webview.create_window(
            'DeepFak3r Voice App',
            'out/index.html',
            js_api=api,
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            shadow=True,
            on_top=False,
            confirm_close=True
        )
        
        # Set window reference for API
        api.set_window(window)
        
        # Register close event
        window.events.closing += on_closing
        
        # Start the webview
        logger.info("Starting webview application...")
        webview.start(debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        exit(1)