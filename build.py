import os
import platform
import subprocess
import sys

def build_app():
    app_name = 'DeepFak3rVoiceApp'
    icon = ''
    windowed = '--windowed'  # No console for GUI app
    
    if platform.system() == 'Darwin':  # macOS
        icon = '--icon app.icns'
        build_cmd = [
            'pyinstaller',
            '--onefile',
            windowed,
            '--name', app_name,
            icon,
            '--add-data', 'out:out',
            'main.py'
        ]
        # For DMG, run additional steps manually (see notes below)
    elif platform.system() == 'Windows':
        icon = '--icon app.ico'
        build_cmd = [
            'pyinstaller',
            '--onefile',
            windowed,
            '--name', app_name,
            icon,
            '--add-data', 'out;out',
            'main.py'
        ]
    else:
        print("Unsupported platform. Only macOS and Windows are supported.")
        return 1
    
    print(f"Building {app_name} for {platform.system()}...")
    subprocess.check_call(build_cmd)
    
    # Clean up
    for dir in ['build', 'dist']:
        if os.path.exists(dir):
            import shutil
            shutil.rmtree(dir)
    
    print(f"Build complete. Check the 'dist/' directory for {app_name}.")
    if platform.system() == 'Darwin':
        print("To create a DMG: brew install create-dmg, then run the DMG creation script separately.")
    return 0

if __name__ == '__main__':
    sys.exit(build_app())