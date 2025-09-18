"""System tray utility and hotkey bindings for the assistant."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(slots=True)
class HotkeyBinding:
    """Store the handle for a keyboard hotkey registration."""

    hotkey: str
    handle: object


class TrayApplication:
    """Manage the pystray icon and global hotkey."""

    def __init__(self, on_activate: Callable[[], None], hotkey: str = "ctrl+shift+space") -> None:
        self._on_activate = on_activate
        self._hotkey = hotkey
        self._icon = None
        self._icon_thread: Optional[threading.Thread] = None
        self._hotkey_binding: Optional[HotkeyBinding] = None

    def start(self) -> None:
        from PIL import Image, ImageDraw
        from pystray import Icon, Menu, MenuItem

        image = Image.new("RGB", (64, 64), color=(40, 40, 40))
        draw = ImageDraw.Draw(image)
        draw.rectangle((16, 16, 48, 48), fill=(120, 200, 255))
        menu = Menu(MenuItem("Activate", lambda: self._on_activate()), MenuItem("Quit", self.stop))
        self._icon = Icon("companion-ai", image, "Companion AI", menu=menu)
        self._icon_thread = threading.Thread(target=self._icon.run, daemon=True)
        self._icon_thread.start()
        self._register_hotkey()

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
        self._unregister_hotkey()

    def _register_hotkey(self) -> None:
        import keyboard

        handle = keyboard.add_hotkey(self._hotkey, self._on_activate)
        self._hotkey_binding = HotkeyBinding(hotkey=self._hotkey, handle=handle)

    def _unregister_hotkey(self) -> None:
        if not self._hotkey_binding:
            return
        import keyboard

        keyboard.remove_hotkey(self._hotkey_binding.handle)
        self._hotkey_binding = None
