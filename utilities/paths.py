"""
Centralized path management for Fluoddity.

This module provides consistent paths for:
- Application directory (bundled files, read-only)
- User data directory (Documents/Fluoddity, user-specific files)

User data is stored in Documents/Fluoddity so users can update the app
by replacing the exe folder without losing their settings.
"""

import sys
import os
import shutil
from pathlib import Path


def get_app_dir() -> Path:
    """Get the application installation directory (where bundled files are)."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return Path(sys.executable).parent
    else:
        # Running from source - project root
        return Path(__file__).parent.parent


def get_user_data_dir() -> Path:
    """Get the user data directory (Documents/Fluoddity)."""
    documents = Path(os.path.expanduser("~")) / "Documents"
    return documents / "Fluoddity"


def ensure_user_data_dir() -> Path:
    """Ensure user data directory exists and return it."""
    user_dir = get_user_data_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_preferences_path() -> Path:
    """Get path to user's preferences.config."""
    return get_user_data_dir() / "preferences.config"


def get_user_keyboard_controls_path() -> Path:
    """Get path to user's keyboard_controls.json."""
    return get_user_data_dir() / "keyboard_controls.json"


def get_imgui_ini_path() -> Path:
    """Get path to imgui.ini (stays in app directory since imgui_bundle doesn't expose ini_filename)."""
    return get_app_dir() / "imgui.ini"


def get_user_physics_configs_dir() -> Path:
    """Get path to user's physics_configs directory (for user-created saves)."""
    return get_user_data_dir() / "physics_configs"


def get_app_physics_configs_dir() -> Path:
    """Get path to bundled physics_configs directory (Core/Advanced)."""
    return get_app_dir() / "physics_configs"


def get_screenshots_dir() -> Path:
    """Get path to Screenshots directory."""
    return get_user_data_dir() / "Screenshots"


def get_videos_dir() -> Path:
    """Get path to Videos directory."""
    return get_user_data_dir() / "Videos"


DEFAULT_ARCHIVE = "default"


def get_archives_root(user_dir=None) -> Path:
    """Root of the named exploration archives (Documents/Fluoddity/archives).

    Each subdirectory is one archive. There is no registry file: the filesystem
    is the list, so an archive folder copied in from elsewhere just appears.
    """
    base = Path(user_dir) if user_dir is not None else get_user_data_dir()
    return base / "archives"


def migrate_legacy_archive(user_dir=None) -> Path:
    """Move the legacy single archive under archives/default.

    A move, not a copy: archives run to hundreds of megabytes of thumbnails.

    If archives/ already exists this does NOTHING - including to a legacy
    archive/ folder sitting beside it. A user who has already migrated and then
    restores an old backup should not have that backup swallowed into a
    directory that already has contents.

    -> the archives root, which is guaranteed to contain a 'default'.
    """
    base = Path(user_dir) if user_dir is not None else get_user_data_dir()
    root = get_archives_root(base)
    if root.exists():
        return root

    legacy = base / "archive"
    if legacy.is_dir():
        try:
            root.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, root / DEFAULT_ARCHIVE)
            print(f"[Fluoddity] archive moved to {root / DEFAULT_ARCHIVE}")
            return root
        except OSError as exc:
            # Launching matters more than migrating. The old folder is left
            # exactly where it is, so nothing is lost.
            print(f"[Fluoddity] could not move the old archive ({exc}); "
                  f"starting an empty '{DEFAULT_ARCHIVE}'")

    (root / DEFAULT_ARCHIVE / "thumbs").mkdir(parents=True, exist_ok=True)
    return root


def get_default_keyboard_controls_path() -> Path:
    """Get path to bundled default_keyboard_controls.json."""
    return get_app_dir() / "default_keyboard_controls.json"


def get_default_imgui_ini_path() -> Path:
    """Get path to bundled default_imgui.ini."""
    return get_app_dir() / "default_imgui.ini"


def initialize_user_data():
    """
    Initialize user data directory on first run.
    Call this once at application startup, before loading any configs.

    Creates the directory structure and copies default files if they don't exist.
    """
    user_dir = ensure_user_data_dir()

    # Create subdirectories
    get_user_physics_configs_dir().mkdir(exist_ok=True)
    get_screenshots_dir().mkdir(exist_ok=True)
    get_videos_dir().mkdir(exist_ok=True)
    migrate_legacy_archive()

    # Copy default keyboard controls if user's doesn't exist
    user_keyboard = get_user_keyboard_controls_path()
    if not user_keyboard.exists():
        default_keyboard = get_default_keyboard_controls_path()
        if default_keyboard.exists():
            shutil.copy(default_keyboard, user_keyboard)
            print(f"[Fluoddity] Created keyboard controls from defaults: {user_keyboard}")

    # Copy default imgui.ini to app directory if it doesn't exist
    # (imgui_bundle doesn't expose ini_filename, so imgui.ini must stay in app dir)
    imgui_ini = get_imgui_ini_path()
    if not imgui_ini.exists():
        default_imgui = get_default_imgui_ini_path()
        if default_imgui.exists():
            shutil.copy(default_imgui, imgui_ini)
            print(f"[Fluoddity] Created imgui.ini from defaults: {imgui_ini}")

    print(f"[Fluoddity] User data directory: {user_dir}")
