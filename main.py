import base64
import mimetypes
import sys
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
from typing import List, Optional, Dict, Any
import atexit
from pydub import AudioSegment
from pydub.effects import normalize
import base64
from appdirs import user_data_dir

APP_NAME = "DeepFak3rVibeVoice"
APP_AUTHOR = "Spruce Emmanuel"

data_path = user_data_dir(APP_NAME, APP_AUTHOR)

# Make sure it exists
os.makedirs(data_path, exist_ok=True)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # When running from PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)



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

class AudioProcessor:
    """Handles advanced audio processing operations"""
    
    @staticmethod
    def base64_to_audio(base64_data: str, format: str = "wav") -> AudioSegment:
        """Convert base64 encoded audio to AudioSegment"""
        audio_bytes = base64.b64decode(base64_data)
        audio_io = io.BytesIO(audio_bytes)
        return AudioSegment.from_file(audio_io, format=format)
    
    @staticmethod
    def audio_to_base64(audio: AudioSegment, format: str = "wav") -> str:
        """Convert AudioSegment to base64 encoded string"""
        buffer = io.BytesIO()
        audio.export(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def export_audio(
        self,
        main_audio: str,
        format: str = "wav",
        bitrate: int = 192,
        sample_rate: int = 44100,
        background_music: Optional[str] = None,
        background_volume: float = 0.3,
        intro: Optional[str] = None,
        outro: Optional[str] = None,
        output_filename: str = "output.wav"
    ) -> Dict[str, Any]:
        """
        Export audio with advanced options including mixing and format conversion
        
        Args:
            main_audio: Base64 encoded main audio
            format: Output format (wav, mp3, ogg)
            bitrate: Bitrate for compressed formats (kbps)
            sample_rate: Sample rate in Hz
            background_music: Base64 encoded background music (optional)
            background_volume: Volume level for background music (0.0 to 1.0)
            intro: Base64 encoded intro audio (optional)
            outro: Base64 encoded outro audio (optional)
            output_filename: Name for the output file
            
        Returns:
            Dict with success status and file path or error message
        """
        try:
            # Load main audio
            main = self.base64_to_audio(main_audio, "webm")
            
            # Set sample rate
            main = main.set_frame_rate(sample_rate)
            
            # Add intro if provided
            if intro:
                intro_audio = self.base64_to_audio(intro)
                intro_audio = intro_audio.set_frame_rate(sample_rate)
                main = intro_audio + main
            
            # Add outro if provided
            if outro:
                outro_audio = self.base64_to_audio(outro)
                outro_audio = outro_audio.set_frame_rate(sample_rate)
                main = main + outro_audio
            
            # Mix with background music if provided
            if background_music:
                bg = self.base64_to_audio(background_music)
                bg = bg.set_frame_rate(sample_rate)
                
                # Adjust background volume
                bg = bg - (20 * (1 - background_volume))  # Reduce volume
                
                # Loop background music if it's shorter than main audio
                if len(bg) < len(main):
                    loops_needed = (len(main) // len(bg)) + 1
                    bg = bg * loops_needed
                
                # Trim background to match main audio length
                bg = bg[:len(main)]
                
                # Mix the audio
                main = main.overlay(bg)
            
            # Normalize audio
            main = normalize(main)
            
            # Determine output path
            output_dir = os.path.expanduser("~/Downloads")
            output_path = os.path.join(output_dir, output_filename)
            
            # Export with appropriate settings
            if format == "mp3":
                main.export(
                    output_path,
                    format="mp3",
                    bitrate=f"{bitrate}k",
                    parameters=["-q:a", "0"]
                )
            elif format == "ogg":
                main.export(
                    output_path,
                    format="ogg",
                    bitrate=f"{bitrate}k"
                )
            else:  # wav
                main.export(
                    output_path,
                    format="wav"
                )
            
            return {
                "success": True,
                "file_path": output_path,
                "message": f"Audio exported successfully to {output_path}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to export audio: {str(e)}"
            }
    
    def trim_audio(
        self,
        audio_data: str,
        start_time: float,
        end_time: float,
        format: str = "wav"
    ) -> Dict[str, Any]:
        """
        Trim audio to specified time range
        
        Args:
            audio_data: Base64 encoded audio
            start_time: Start time in seconds
            end_time: End time in seconds
            format: Audio format
            
        Returns:
            Dict with success status and trimmed audio data
        """
        try:
            audio = self.base64_to_audio(audio_data, format)
            
            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Trim audio
            trimmed = audio[start_ms:end_ms]
            
            # Convert back to base64
            trimmed_base64 = self.audio_to_base64(trimmed, format)
            
            return {
                "success": True,
                "audio_data": trimmed_base64,
                "duration": len(trimmed) / 1000.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def convert_format(
        self,
        audio_data: str,
        input_format: str,
        output_format: str,
        bitrate: int = 192
    ) -> Dict[str, Any]:
        """
        Convert audio from one format to another
        
        Args:
            audio_data: Base64 encoded audio
            input_format: Input audio format
            output_format: Desired output format
            bitrate: Bitrate for compressed formats
            
        Returns:
            Dict with success status and converted audio data
        """
        try:
            audio = self.base64_to_audio(audio_data, input_format)
            
            # Export to new format
            buffer = io.BytesIO()
            if output_format == "mp3":
                audio.export(buffer, format="mp3", bitrate=f"{bitrate}k")
            elif output_format == "ogg":
                audio.export(buffer, format="ogg", bitrate=f"{bitrate}k")
            else:
                audio.export(buffer, format=output_format)
            
            buffer.seek(0)
            converted_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return {
                "success": True,
                "audio_data": converted_base64,
                "format": output_format
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class HardwareDetector:
    """Detects system hardware with robust multi-vendor GPU support."""
    
    # Performance scores for AI workloads (relative scale 0-100)
    GPU_PERFORMANCE_DB = {
        # NVIDIA RTX 50 Series (2025)
        "rtx 5090": 100, "rtx 5080": 85, "rtx 5070 ti": 79,
        "rtx 5070": 68, "rtx 5060 ti": 54, "rtx 5060": 45,
        
        # NVIDIA RTX 40 Series
        "rtx 4090": 95, "rtx 4080 super": 83, "rtx 4080": 82,
        "rtx 4070 ti super": 74, "rtx 4070 ti": 71, "rtx 4070 super": 67,
        "rtx 4070": 63, "rtx 4060 ti": 55, "rtx 4060": 50,
        
        # NVIDIA RTX 30 Series
        "rtx 3090 ti": 88, "rtx 3090": 85, "rtx 3080 ti": 80,
        "rtx 3080": 75, "rtx 3070 ti": 65, "rtx 3070": 60,
        "rtx 3060 ti": 52, "rtx 3060": 45,
        
        # AMD RX 9000 Series (2025)
        "rx 9070 xt": 80, "rx 9070": 75, "rx 9060 xt": 60,
        
        # AMD RX 7000/8000 Series
        "rx 7900 xtx": 85, "rx 7900 xt": 80, "rx 7900 gre": 75,
        "rx 7800 xt": 70, "rx 7700 xt": 60, "rx 7600 xt": 55, "rx 7600": 48,
        
        # AMD RX 6000 Series
        "rx 6900 xt": 78, "rx 6800 xt": 72, "rx 6800": 68,
        "rx 6700 xt": 58, "rx 6600 xt": 48, "rx 6600": 42,
        
        # Intel Arc Battlemage (2025)
        "arc b580": 50, "arc b570": 45, "arc b560": 40,
        
        # Intel Arc Alchemist
        "arc a770": 52, "arc a750": 48, "arc a580": 42,
        
        # Apple Silicon (unified memory architecture)
        "apple m4 max": 75, "apple m4 pro": 70, "apple m4": 60,
        "apple m3 max": 72, "apple m3 pro": 65, "apple m3": 55,
        "apple m2 ultra": 80, "apple m2 max": 70, "apple m2 pro": 65, "apple m2": 50,
        "apple m1 ultra": 75, "apple m1 max": 68, "apple m1 pro": 60, "apple m1": 50,
    }
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect system hardware and return comprehensive information."""
        os_name, arch = self._get_os_arch()
        logger.info(f"Detecting hardware on {os_name}/{arch}")
        
        gpus = []
        
        try:
            if os_name in ["windows", "linux"]:
                gpus.extend(self._detect_nvidia_gpu())
                gpus.extend(self._detect_amd_gpu())
                gpus.extend(self._detect_intel_gpu())
            elif os_name == "macos":
                gpus.extend(self._detect_apple_silicon())
        except Exception as e:
            logger.error(f"Hardware detection error: {e}", exc_info=True)
        
        # Remove duplicates (same model detected by multiple methods)
        gpus = self._deduplicate_gpus(gpus)
        
        # Calculate totals
        total_vram = sum(gpu["vramGB"] for gpu in gpus)
        has_gpu = len(gpus) > 0
        
        # Determine recommended setup based on capabilities
        recommended_setup = self._determine_setup(has_gpu, total_vram, gpus)
        
        result = {
            "os": os_name,
            "arch": arch,
            "hasGPU": has_gpu,
            "gpus": gpus,
            "totalVRAM": round(total_vram, 1),
            "recommendedSetup": recommended_setup
        }
        
        logger.info(f"Hardware detection complete: {len(gpus)} GPU(s), {total_vram:.1f}GB VRAM")
        return result
    
    def _detect_nvidia_gpu(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        gpus = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if ',' in line:
                        name, vram_mb = line.split(',', 1)
                        model = self._normalize_gpu_name(name.strip(), "nvidia")
                        
                        try:
                            vram_gb = float(vram_mb.strip()) / 1024
                        except ValueError:
                            logger.warning(f"Could not parse VRAM for {name}")
                            continue
                        
                        perf_score = self._get_performance_score(model)
                        
                        gpus.append({
                            "vendor": "nvidia",
                            "model": model,
                            "vramGB": round(vram_gb, 1),
                            "isSupported": vram_gb >= 6,
                            "performanceScore": perf_score
                        })
                        logger.debug(f"Detected NVIDIA GPU: {model} ({vram_gb:.1f}GB)")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
        except Exception as e:
            logger.error(f"NVIDIA detection error: {e}")
        
        return gpus
    
    def _detect_amd_gpu(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using rocm-smi."""
        gpus = []
        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.split('\n')
                current_gpu = {}
                
                for line in lines:
                    line_lower = line.lower()
                    
                    # Extract GPU name
                    if 'gpu' in line_lower and 'product name' in line_lower and ':' in line:
                        model_text = line.split(':', 1)[1].strip()
                        model = self._normalize_gpu_name(model_text, "amd")
                        current_gpu['model'] = model
                    
                    # Extract VRAM
                    if 'vram total' in line_lower or 'memory total' in line_lower:
                        vram_match = re.search(r'(\d+)\s*(mb|gb)', line_lower)
                        if vram_match:
                            vram_value = float(vram_match.group(1))
                            unit = vram_match.group(2)
                            vram_gb = vram_value / 1024 if unit == 'mb' else vram_value
                            current_gpu['vram'] = vram_gb
                    
                    # When we have both model and VRAM, add the GPU
                    if 'model' in current_gpu and 'vram' in current_gpu:
                        perf_score = self._get_performance_score(current_gpu['model'])
                        gpus.append({
                            "vendor": "amd",
                            "model": current_gpu['model'],
                            "vramGB": round(current_gpu['vram'], 1),
                            "isSupported": current_gpu['vram'] >= 6,
                            "performanceScore": perf_score
                        })
                        logger.debug(f"Detected AMD GPU: {current_gpu['model']} ({current_gpu['vram']:.1f}GB)")
                        current_gpu = {}  # Reset for next GPU
                        
        except FileNotFoundError:
            logger.debug("rocm-smi not found")
        except subprocess.TimeoutExpired:
            logger.warning("rocm-smi timed out")
        except Exception as e:
            logger.error(f"AMD detection error: {e}")
        
        return gpus
    
    def _detect_intel_gpu(self) -> List[Dict[str, Any]]:
        """Detect Intel Arc GPUs."""
        gpus = []
        
        # Try xpu-smi first (newer Intel tool)
        try:
            result = subprocess.run(
                ['xpu-smi', 'discovery'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'arc' in result.stdout.lower():
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'device name' in line.lower() and ':' in line:
                        model_text = line.split(':', 1)[1].strip()
                        model = self._normalize_gpu_name(model_text, "intel")
                        
                        # Try to find VRAM in nearby lines
                        vram_gb = 8.0  # Default for Arc cards
                        for j in range(max(0, i-3), min(len(lines), i+4)):
                            vram_match = re.search(r'(\d+)\s*(gb|mb)', lines[j].lower())
                            if vram_match and 'memory' in lines[j].lower():
                                vram_value = float(vram_match.group(1))
                                unit = vram_match.group(2)
                                vram_gb = vram_value / 1024 if unit == 'mb' else vram_value
                                break
                        
                        perf_score = self._get_performance_score(model)
                        gpus.append({
                            "vendor": "intel",
                            "model": model,
                            "vramGB": round(vram_gb, 1),
                            "isSupported": vram_gb >= 6,
                            "performanceScore": perf_score
                        })
                        logger.debug(f"Detected Intel GPU: {model} ({vram_gb:.1f}GB)")
        except FileNotFoundError:
            logger.debug("xpu-smi not found, trying clinfo")
        except Exception as e:
            logger.debug(f"xpu-smi detection failed: {e}")
        
        # Fallback to clinfo
        if not gpus:
            try:
                result = subprocess.run(
                    ['clinfo'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and 'intel' in result.stdout.lower():
                    model_match = re.search(r'intel.*arc\s+([^\n]+)', result.stdout.lower())
                    if model_match:
                        model = self._normalize_gpu_name(f"arc {model_match.group(1)}", "intel")
                        vram_gb = 8.0  # Default
                        
                        perf_score = self._get_performance_score(model)
                        gpus.append({
                            "vendor": "intel",
                            "model": model,
                            "vramGB": vram_gb,
                            "isSupported": True,
                            "performanceScore": perf_score
                        })
                        logger.debug(f"Detected Intel GPU via clinfo: {model}")
            except FileNotFoundError:
                logger.debug("clinfo not found")
            except Exception as e:
                logger.debug(f"clinfo detection failed: {e}")
        
        return gpus
    
    def _detect_apple_silicon(self) -> List[Dict[str, Any]]:
        """Detect Apple Silicon chips."""
        gpus = []
        try:
            # Get chip model
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                chip_name = result.stdout.strip().lower()
                model = self._normalize_gpu_name(chip_name, "apple")
                
                # Get total system memory (unified memory architecture)
                mem_result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                vram_gb = 8.0  # Default
                if mem_result.returncode == 0:
                    try:
                        mem_bytes = int(mem_result.stdout.strip())
                        vram_gb = mem_bytes / (1024 ** 3)
                    except ValueError:
                        pass
                
                perf_score = self._get_performance_score(model)
                gpus.append({
                    "vendor": "apple",
                    "model": model,
                    "vramGB": round(vram_gb, 1),
                    "isSupported": True,  # Apple Silicon always supported
                    "performanceScore": perf_score
                })
                logger.debug(f"Detected Apple Silicon: {model} ({vram_gb:.1f}GB unified memory)")
                
        except Exception as e:
            logger.error(f"Apple Silicon detection error: {e}")
        
        return gpus
    
    def _normalize_gpu_name(self, name: str, vendor: str) -> str:
        """Normalize GPU model names for consistent matching."""
        name = name.lower().strip()
        
        # Remove common prefixes
        prefixes = [
            'nvidia', 'geforce', 'amd', 'radeon', 'intel', 'apple', 
            'graphics', 'gpu', 'chip'
        ]
        for prefix in prefixes:
            name = re.sub(rf'\b{prefix}\b', '', name, flags=re.IGNORECASE)
        
        # Clean up whitespace
        name = ' '.join(name.split())
        
        # Vendor-specific normalization
        if vendor == "amd":
            # "Radeon RX 7900 XTX" -> "rx 7900 xtx"
            name = re.sub(r'\brx\s*', 'rx ', name)
        elif vendor == "intel":
            # Ensure "arc" prefix
            if 'arc' not in name and any(x in name for x in ['a770', 'a750', 'a580', 'b580', 'b570', 'b560']):
                name = 'arc ' + name
        elif vendor == "apple":
            # Extract M-series chip version
            m_match = re.search(r'm\d+\s*(ultra|max|pro)?', name)
            if m_match:
                name = f"apple {m_match.group(0).strip()}"
            else:
                name = f"apple {name}"
        
        return name.strip()
    
    def _get_performance_score(self, model: str) -> int:
        """Get performance score for a GPU model."""
        # Try exact match first
        if model in self.GPU_PERFORMANCE_DB:
            return self.GPU_PERFORMANCE_DB[model]
        
        # Try fuzzy matching for close variants
        model_base = model.split()[0] if ' ' in model else model
        for db_model, score in self.GPU_PERFORMANCE_DB.items():
            if model_base in db_model or db_model in model:
                return score
        
        # Default score for unknown GPUs
        return 50
    
    def _deduplicate_gpus(self, gpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate GPU entries."""
        seen = set()
        unique_gpus = []
        
        for gpu in gpus:
            key = (gpu["vendor"], gpu["model"], gpu["vramGB"])
            if key not in seen:
                seen.add(key)
                unique_gpus.append(gpu)
        
        return unique_gpus
    
    def _determine_setup(self, has_gpu: bool, total_vram: float, gpus: List[Dict[str, Any]]) -> str:
        """Determine recommended AI setup based on hardware."""
        if not has_gpu:
            return "runpod"  # No GPU -> cloud recommended
        
        # Check if any GPU is high-performance
        max_perf = max((gpu["performanceScore"] for gpu in gpus), default=0)
        
        if total_vram >= 16 and max_perf >= 60:
            return "local"  # Strong GPU -> local recommended
        elif total_vram >= 8 and max_perf >= 45:
            return "local"  # Decent GPU -> local possible
        elif total_vram >= 6:
            return "runpod"  # Weak GPU -> cloud recommended
        else:
            return "runpod"  # Insufficient VRAM
    
    def _get_os_arch(self) -> tuple[str, str]:
        """Get normalized OS and architecture."""
        os_name = platform.system().lower()
        arch = platform.machine().lower()

        # Normalize OS name
        if os_name == "darwin":
            os_name = "macos"
        elif os_name not in ["windows", "linux"]:
            os_name = "linux"  # Default fallback

        # Detect real arch on Apple Silicon Macs (even under Rosetta)
        if os_name == "macos" and arch == "x86_64":
            try:
                # sysctl check: if returns 1, it's running under Rosetta on Apple Silicon
                output = subprocess.check_output(["sysctl", "-in", "sysctl.proc_translated"])
                if output.strip() == b"1":
                    arch = "arm64"
            except Exception:
                pass

        # Normalize architecture
        if arch in ["x86_64", "amd64"]:
            arch = "x64"
        elif arch in ["arm64", "aarch64"]:
            arch = "arm64"
        elif arch in ["i386", "i686"]:
            arch = "x86"
        else:
            arch = "x64"  # Default fallback

        return os_name, arch

class RunPodManager:
    """Manages RunPod deployments from the Python backend."""
    def __init__(self, api_key: str, progress_callback):
        self.api_key = api_key
        self.progress_callback = progress_callback
        self.base_url = "https://api.runpod.io/graphql"

    def _graphql_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Performs a GraphQL request to the RunPod API."""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={"query": query, "variables": variables},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if "errors" in data:
                # Log the full error details
                logger.error(f"RunPod GraphQL errors: {json.dumps(data['errors'], indent=2)}")
                raise Exception(f"RunPod GraphQL error: {data['errors'][0]['message']}")
            return data.get("data", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"RunPod API request failed: {e}")
            # Log response body if available
            if hasattr(e.response, 'text'):
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def validate_api_key(self) -> bool:
        """Validates the RunPod API key."""
        query = """
        query {
            myself {
                id
                clientBalance
            }
        }
        """
        try:
            data = self._graphql_request(query)
            balance = data.get("myself", {}).get("currentBalance", 0)
            logger.info(f"RunPod account balance: ${balance}")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    def get_available_gpus(self):
        """Fetches available GPU types suitable for the TTS server."""
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice }
            }
        }
        """
        gpus = self._graphql_request(query).get("gpuTypes", [])
        
        # Filter for GPUs with >= 24GB VRAM, secure cloud, and in stock
        suitable_gpus = [
            gpu for gpu in gpus
            if gpu.get("memoryInGb", 0) >= 24
            and gpu.get("secureCloud")
            
        ]
        return suitable_gpus

    def _select_optimal_gpu(self, gpu_types):
        """Selects the best GPU based on a scoring system (cost/performance)."""
        return min(gpu_types, key=lambda gpu: gpu['lowestPrice']['uninterruptablePrice'])

    def deploy_tts_server(self, instance_name: str) -> Dict[str, Any]:
        """Orchestrates the deployment of the TTS server on RunPod."""
        try:
            self.progress_callback({
                "stage": "validating", "progress": 10, "message": "Validating API key..."
            })
            if not self.validate_api_key():
                raise ValueError("Invalid RunPod API key or insufficient credits")

            self.progress_callback({
                "stage": "validating", "progress": 20, "message": "Finding available GPUs..."
            })
            gpu_types = self.get_available_gpus()
            if not gpu_types:
                raise RuntimeError("No suitable GPU types available (24GB+ VRAM required)")

            selected_gpu = self._select_optimal_gpu(gpu_types)
            self.progress_callback({
                "stage": "creating", "progress": 40,
                "message": f"Deploying on {selected_gpu['displayName']}..."
            })

            # Create Pod
            create_mutation = """
            mutation createPod($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    imageName
                    machine {
                        podHostId
                    }
                }
            }
            """
            variables = {
                "input": {
                    "cloudType": "SECURE",
                    "gpuCount": 1,
                    "volumeInGb": 100,
                    "containerDiskInGb": 50,
                    "minVcpuCount": 4,
                    "minMemoryInGb": 16,
                    "gpuTypeId": selected_gpu["id"],
                    "name": instance_name,
                    "imageName": "spruceemma/vibevoice-server:latest",
                    "dockerArgs": "",
                    "ports": "8000/tcp",
                    "volumeMountPath": "/workspace",
                }
            }
            result = self._graphql_request(create_mutation, variables)
            pod_id = result["podFindAndDeployOnDemand"]["id"]

            server_url = self._wait_for_server_ready(pod_id)

            self.progress_callback({
                "stage": "ready", "progress": 100, "message": "Server is ready!",
                "podId": pod_id, "url": server_url
            })
            return {"podId": pod_id, "url": server_url}

        except Exception as e:
            logger.error(f"RunPod deployment failed: {e}", exc_info=True)
            self.progress_callback({
                "stage": "error", "progress": 0, "message": "Deployment failed",
                "error": str(e)
            })
            raise

    def _wait_for_server_ready(self, pod_id: str, timeout=600) -> str:
            """Waits for the pod to become ready and return its public URL."""
            start_time = time.time()
            get_pod_query = """
            query getPod($podId: String!) {
                pod(input: {podId: $podId}) {
                    runtime {
                        ports { ip isIpPublic privatePort publicPort }
                    }
                }
            }
            """
            
            while time.time() - start_time < timeout:
                progress = min(60 + int((time.time() - start_time) / timeout * 35), 95)
                self.progress_callback({
                    "stage": "starting", "progress": progress, "podId": pod_id,
                    "message": f"Waiting for server to start... ({int(time.time() - start_time)}s)"
                })
                
                pod_data = self._graphql_request(get_pod_query, {"podId": pod_id})
                logging.info(f"Pod data: {json.dumps(pod_data, indent=2)}")
                
                if pod_data is None:
                    logger.warning(f"Pod data is None for pod {pod_id}, retrying...")
                    time.sleep(5)
                    continue
                
                pod_obj = pod_data.get("pod")
                if pod_obj is None:
                    logger.info(f"Pod {pod_id} not yet available, retrying...")
                    time.sleep(5)
                    continue
                
                # Safely get the runtime object
                runtime_info = pod_obj.get("runtime")
                
                # Check if runtime info is available yet. If not, continue waiting.
                if not runtime_info:
                    logger.info(f"Pod {pod_id} runtime not yet initialized, retrying...")
                    time.sleep(5)
                    continue

                # Now we can safely get the ports
                ports = runtime_info.get("ports", [])
                
                if ports:
                    http_port = next((p for p in ports if p.get("privatePort") == 8000 and p.get("isIpPublic")), None)
                    if http_port:
                        url = f"http://{http_port['ip']}:{http_port['publicPort']}"
                        try:
                            response = requests.get(f"{url}/health", timeout=5)
                            if response.ok:
                                logger.info(f"RunPod server is healthy at {url}")
                                return url
                        except requests.RequestException:
                            pass # Ignore connection errors while waiting
                
                time.sleep(5)
                
            raise TimeoutError("Server failed to become ready within the timeout period.")

class LocalServerManager:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        self.server_process = None
        self.log_file = None
        self.setup_cancelled = False
        self._lock = threading.Lock()
        
        # Configuration
        self.github_repo = "iamspruce/deepfak3voice"  
        self.release_tag = "latest"
        self.server_port = 8000
        self.os_name, self.arch = self._get_os_arch()
        self.install_path = self._get_applications_folder(self.os_name)
        self.executable_name = "vibevoice-server.exe" if self.os_name == "windows" else "vibevoice-server"
        self.executable_path = os.path.join(self.install_path, self.executable_name)
        self.config_path = os.path.join(self.install_path, "config.json")
        self.log_path = os.path.join(self.install_path, "server.log")
        
        # Register cleanup on exit
        atexit.register(self._cleanup_resources)
        
    def cancel_setup(self):
        """Cancel ongoing setup."""
        with self._lock:
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
            arch = "amd64"
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
            install_dir = os.path.join(base_dir, 'DeepFak3rVibeVoiceServer')
        elif os_name == "macos":
            install_dir = os.path.expanduser('~/Applications/DeepFak3rVibeVoiceServer')
        else:  # linux
            install_dir = os.path.expanduser('~/.local/share/deepfak3r-vibevoice-server')
        
        # Ensure directory exists and is writable
        Path(install_dir).mkdir(parents=True, exist_ok=True)
        
        if not os.access(install_dir, os.W_OK):
            raise PermissionError(f"No write access to {install_dir}")
            
        return install_dir

    def _get_download_url(self, os_name, arch):
        """Get download URL with fallback options."""
        base_url = f"https://github.com/{self.github_repo}/releases/latest/download"
        filename = f"vibevoice-{os_name}-{arch}"
        ext = "zip" if os_name == "windows" else "tar.gz"
        return f"{base_url}/{filename}.{ext}", f"{filename}.{ext}"

    def _verify_download(self, file_path, expected_checksum=None):
        """Verify downloaded file integrity."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Downloaded file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            raise ValueError(f"Downloaded file too small: {file_size} bytes")
            
        # Checksum verification if provided
        if expected_checksum:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != expected_checksum:
                    raise ValueError(f"Checksum mismatch for {file_path}")
        
        logger.info(f"Download verified: {file_path} ({file_size:,} bytes)")

    def setup_server(self):
        """Enhanced server setup with comprehensive error handling."""
        try:
            with self._lock:
                if self.setup_cancelled:
                    logger.info("Setup cancelled before starting")
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
            
            with self._lock:
                if self.setup_cancelled:
                    self._cleanup_partial_download(archive_path)
                    return

            # Verify download
            self._verify_download(archive_path)

            # Extract
            self._extract_with_progress(archive_path, self.install_path)
            
            with self._lock:
                if self.setup_cancelled:
                    self._cleanup_partial_download(archive_path)
                    return

            # Find the executable after extraction
            self._locate_executable()

            # Cleanup archive
            self._safe_remove(archive_path)

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

    def _cleanup_partial_download(self, archive_path):
        """Clean up partially downloaded files."""
        logger.info("Cleaning up partial download")
        self._safe_remove(archive_path)
        self.progress_callback({
            "stage": "error",
            "progress": 0,
            "message": "Setup cancelled by user"
        })

    def _safe_remove(self, file_path):
        """Safely remove a file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {e}")

    def _locate_executable(self):
        """Find the executable after extraction (handles subdirectories)."""
        # First check if it's directly in install_path
        if os.path.exists(self.executable_path):
            logger.info(f"Executable found at: {self.executable_path}")
            return

        # Search in subdirectories
        for root, dirs, files in os.walk(self.install_path):
            if self.executable_name in files:
                self.executable_path = os.path.join(root, self.executable_name)
                logger.info(f"Executable found at: {self.executable_path}")
                return
        
        raise FileNotFoundError(f"Executable '{self.executable_name}' not found after extraction")

    def check_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status information."""
        with self._lock:
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
        """Download with progress reporting and cancellation support."""
        self.progress_callback({
            "stage": "downloading",
            "progress": 0,
            "message": "Starting download..."
        })

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            content_length = response.headers.get('content-length')
            total_size = int(content_length) if content_length else 0

            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    with self._lock:
                        if self.setup_cancelled:
                            logger.info("Download cancelled")
                            return
                        
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 80)
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
        """Extract archive with progress reporting and cancellation support."""
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
                        with self._lock:
                            if self.setup_cancelled:
                                logger.info("Extraction cancelled")
                                return
                            
                        zip_ref.extract(member, extract_path)
                        progress = 80 + int((i + 1) / total_files * 10)
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
                        with self._lock:
                            if self.setup_cancelled:
                                logger.info("Extraction cancelled")
                                return
                            
                        tar_ref.extract(member, extract_path)
                        progress = 80 + int((i + 1) / total_files * 10)
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
        with self._lock:
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
                startup_flags['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
            else:
                startup_flags['close_fds'] = True
                
            # Start server process
            cmd = [self.executable_path, "--host", "0.0.0.0", "--port", str(self.server_port)]
            
            # Close existing log file if open
            self._close_log_file()
            
            # Open new log file
            self.log_file = open(self.log_path, 'w')
            
            with self._lock:
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=self.log_file,
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
            self._close_log_file()
            self.progress_callback({
                "stage": "error",
                "progress": 0,
                "message": "Failed to start server",
                "error": str(e)
            })

    def stop_server(self):
        """Stop the running server."""
        with self._lock:
            if not self.server_process:
                return
            
            process = self.server_process
            self.server_process = None
        
        try:
            process.terminate()
            process.wait(timeout=10)
            logger.info("Local server stopped gracefully")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.info("Local server forcefully stopped")
        except Exception as e:
            logger.error(f"Failed to stop local server: {e}")
        
        # Close log file
        self._close_log_file()

    def _close_log_file(self):
        """Safely close the log file."""
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception as e:
                logger.warning(f"Failed to close log file: {e}")

    def _cleanup_resources(self):
        """Cleanup resources on exit."""
        self.stop_server()

    def _kill_existing_server(self):
        """Kill any existing server processes on the target port."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    f'netstat -ano | findstr :{self.server_port}',
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'LISTENING' in line:
                            parts = line.split()
                            if parts:
                                pid = parts[-1]
                                try:
                                    subprocess.run(
                                        f'taskkill /PID {pid} /F', 
                                        shell=True,
                                        check=False,
                                        capture_output=True
                                    )
                                    logger.info(f"Killed existing process PID {pid}")
                                except Exception as e:
                                    logger.warning(f"Failed to kill PID {pid}: {e}")
            else:
                try:
                    result = subprocess.run(
                        ['lsof', '-ti', f'tcp:{self.server_port}'],
                        capture_output=True, 
                        text=True
                    )
                    if result.stdout:
                        pids = result.stdout.strip().split('\n')
                        for pid in pids:
                            if pid:
                                try:
                                    subprocess.run(['kill', '-9', pid], check=False)
                                    logger.info(f"Killed existing process PID {pid}")
                                except Exception as e:
                                    logger.warning(f"Failed to kill PID {pid}: {e}")
                except FileNotFoundError:
                    logger.warning("lsof not available, skipping port cleanup")
        except Exception as e:
            logger.warning(f"Failed to kill existing server: {e}")

    def _wait_for_server_ready(self, timeout=600):
        """Wait for server to be ready with health checks (reduced timeout)."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if self.setup_cancelled:
                    return False
                
                # Check if process is still running
                if self.server_process and self.server_process.poll() is not None:
                    logger.error("Server process terminated unexpectedly")
                    return False
            
            try:
                response = requests.get(
                    f'http://localhost:{self.server_port}/health',
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("Server health check passed")
                    return True
                    
            except requests.exceptions.RequestException:
                pass
            
            elapsed = int(time.time() - start_time)
            self.progress_callback({
                "stage": "starting",
                "progress": min(90 + int((elapsed / timeout) * 10), 99),
                "message": f"Waiting for server... ({elapsed}s/{timeout}s)"
            })
            
            time.sleep(2)
        
        logger.error(f"Server failed to become ready within {timeout} seconds")
        return False
         
class Api:
    """PyWebView API class for exposing Python functions to JavaScript."""
    
    def __init__(self):
        self.processor = AudioProcessor()
        self.detector = HardwareDetector()
        self._hardware_info = self.detector.detect_hardware()
        self._active_runpod_instance: Optional[Dict[str, str]] = None
        self.local_server_manager = LocalServerManager(self.send_progress_to_js)
        self.window = None
        self._setup_thread: Optional[threading.Thread] = None
        self._runpod_thread: Optional[threading.Thread] = None

    def set_window(self, window):
        """Set the PyWebView window instance for JS communication."""
        self.window = window
        logger.info("PyWebView window reference set")

    def export_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export audio with advanced options"""
        return self.processor.export_audio(**params)
    
    def trim_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trim audio to specified range"""
        return self.processor.trim_audio(**params)
    
    def convert_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert audio format"""
        return self.processor.convert_format(**params)
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Return cached hardware information (detected at startup)."""
        return self._hardware_info

    def send_progress_to_js(self, progress_data):
        """Enhanced progress reporting with error handling."""
        try:
            if self.window:
                js_code = f'window.dispatchLocalServerProgress({json.dumps(progress_data)})'
                self.window.evaluate_js(js_code)
                logger.debug(f"Progress update sent: {progress_data['stage']} - {progress_data['progress']}%")
        except Exception as e:
            logger.error(f"Failed to send progress to JS: {e}")
            
    def start_local_server_setup(self):
        """
        Thread-safe server setup with better error handling.
        Called by JS via: window.pywebview.api.start_local_server_setup()
        """
        try:
            # Check if setup is already in progress
            if self._setup_thread and self._setup_thread.is_alive():
                logger.warning("Setup already in progress")
                self.send_progress_to_js({
                    "stage": "error",
                    "progress": 0,
                    "message": "Setup is already in progress",
                    "error": "Another setup operation is running"
                })
                return

            # Reset cancellation flag
            self.local_server_manager.setup_cancelled = False

            # Start setup in a new thread
            self._setup_thread = threading.Thread(
                target=self._safe_setup_wrapper,
                daemon=True,
                name="LocalServerSetup"
            )
            self._setup_thread.start()
            logger.info("Local server setup thread started")
            
        except Exception as e:
            logger.error(f"Failed to start setup thread: {e}", exc_info=True)
            self.send_progress_to_js({
                "stage": "error",
                "progress": 0,
                "message": "Failed to start setup",
                "error": str(e)
            })
            
    def _safe_setup_wrapper(self):
        """Wrapper for setup with comprehensive error handling."""
        try:
            logger.info("Starting server setup")
            self.local_server_manager.setup_server()
            logger.info("Server setup completed")
        except Exception as e:
            logger.error(f"Setup failed with exception: {e}", exc_info=True)
            self.send_progress_to_js({
                "stage": "error",
                "progress": 0,
                "message": "Setup failed unexpectedly",
                "error": str(e)
            })

    def cancel_local_server_setup(self):
        """
        Cancel ongoing server setup.
        Called by JS via: window.pywebview.api.cancel_local_server_setup()
        """
        try:
            logger.info("Cancellation requested for server setup")
            
            # Fixed: Use self.local_server_manager instead of self.server_manager
            if self.local_server_manager:
                self.local_server_manager.cancel_setup()
                logger.info("Cancellation signal sent to LocalServerManager")
            else:
                logger.warning("No server manager instance to cancel")
                
        except Exception as e:
            logger.error(f"Error cancelling setup: {e}", exc_info=True)

    def check_local_server(self):
        """
        Called by JS to get the current server status.
        Called by JS via: window.pywebview.api.check_local_server()
        
        Returns:
            dict: Server status matching the LocalServerInfo TypeScript interface
        """
        try:
            info = self.local_server_manager.check_server_status()
            logger.debug(f"Local server status: installed={info['isInstalled']}, running={info['isRunning']}")
            return info
        except Exception as e:
            logger.error(f"Error checking local server: {e}", exc_info=True)
            # Return a default error state if checks fail
            return {
                "isInstalled": False,
                "isRunning": False,
                "port": 8000,
                "version": "unknown",
                "installPath": self.local_server_manager.install_path if self.local_server_manager else "",
                "executablePath": self.local_server_manager.executable_path if self.local_server_manager else "",
                "configPath": self.local_server_manager.config_path if self.local_server_manager else "",
                "logPath": self.local_server_manager.log_path if self.local_server_manager else "",
                "pid": None,
                "lastStarted": None
            }

    def start_local_server(self):
        """
        Called by JS to start the installed local server.
        Called by JS via: window.pywebview.api.start_local_server()
        """
        try:
            logger.info("Start server request received")
            
            # Check if server is already running
            if (self.local_server_manager.server_process and 
                self.local_server_manager.server_process.poll() is None):
                logger.info("Server is already running")
                # Send a progress update to inform the UI
                self.send_progress_to_js({
                    "stage": "complete",
                    "progress": 100,
                    "message": "Server is already running"
                })
                return
            
            # Start the server
            self.local_server_manager.start_server()
            logger.info("Server start request completed")
            
        except Exception as e:
            logger.error(f"Failed to start local server: {e}", exc_info=True)
            self.send_progress_to_js({
                "stage": "error",
                "progress": 0,
                "message": "Failed to start server",
                "error": str(e)
            })
            raise

    def stop_local_server(self):
        """
        Called by JS to stop the running local server.
        Called by JS via: window.pywebview.api.stop_local_server()
        """
        try:
            logger.info("Stop server request received")
            self.local_server_manager.stop_server()
            logger.info("Server stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop local server: {e}", exc_info=True)
            raise

    def send_runpod_progress_to_js(self, progress_data):
        """Sends RunPod deployment progress to the JS frontend."""
        try:
            if self.window:
                # Dispatch a custom event that the React component can listen for
                js_code = f'window.dispatchEvent(new CustomEvent("runpod-progress", {{ detail: {json.dumps(progress_data)} }}));'
                self.window.evaluate_js(js_code)
                logger.info(f"RunPod progress sent: {progress_data.get('stage')} - {progress_data.get('progress')}%")
        except Exception as e:
            logger.error(f"Failed to send RunPod progress to JS: {e}")

    def start_runpod_deployment(self, api_key: str):
        """Starts the RunPod deployment process in a separate thread."""
        if self._runpod_thread and self._runpod_thread.is_alive():
            logger.warning("RunPod deployment already in progress.")
            return

        self._runpod_thread = threading.Thread(
            target=self._safe_runpod_deploy_wrapper,
            args=(api_key,),
            daemon=True
        )
        self._runpod_thread.start()
        logger.info("RunPod deployment thread started.")

    def _safe_runpod_deploy_wrapper(self, api_key: str):
        """Wrapper for RunPod deployment with error handling."""
        try:
            manager = RunPodManager(api_key, self.send_runpod_progress_to_js)
            instance_name = f"vibevoice-server-{int(time.time())}"
            manager.deploy_tts_server(instance_name)
        except Exception as e:
            logger.error(f"RunPod deployment failed in wrapper: {e}", exc_info=True)
            # The manager already sends a detailed error, this is a fallback.
            self.send_runpod_progress_to_js({
                "stage": "error", "progress": 0, "message": "Deployment failed", "error": str(e)
            })
            
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
            
    def cleanup_on_exit(self):
        """
        Cleanup when app is closing
        Called automatically on window close
        """
        print("🧹 Cleaning up TTS API resources...")
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
        index_path = resource_path("out/index.html")

        window = webview.create_window(
            'DeepFak3r VibeVoice',
            index_path,
            js_api=api,
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            shadow=True,
            on_top=False,
            confirm_close=True,

        )
        
        # Set window reference for API
        api.set_window(window)
        
        # Register close event
        window.events.closing += on_closing
        
        # Start the webview
        logger.info("Starting webview application...")
        webview.start(debug=False,private_mode=False,storage_path=data_path)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        exit(1)