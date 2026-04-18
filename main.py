"""
main.py
───────
Entry point for FrameForge — Auto-Label & Pseudo-Label Studio.

Responsibilities
----------------
1. Instantiate the QApplication.
2. Apply the ``Fusion`` style and a custom dark QPalette using the exact
   colour values from the design brief.
3. Set the global application font (Ubuntu 9pt on Linux/macOS,
   Segoe UI 9pt on Windows) so that every widget inherits it.
4. Construct and show the MainWindow, then enter the Qt event loop.
"""

import platform
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

from gui import MainWindow


# ─────────────────────────────────────────────────────────────────────────── #
#  Theme                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def build_dark_palette() -> QPalette:
    """
    Construct and return a dark QPalette using the exact colour values
    specified in the FrameForge design brief.

    Colour reference
    ----------------
    Window / Button background : ``#2D2D30``
    Base (input backgrounds)   : ``#1E1E1E``
    Text / WindowText          : ``#DCDCDC``
    Highlight (selection)      : ``#2A82DA``
    Mid / borders              : ``#464649``

    Returns
    -------
    QPalette
        Fully configured dark palette ready to be applied to QApplication.
    """
    palette = QPalette()

    # ── Core colours ────────────────────────────────────────────────────
    bg        = QColor("#2D2D30")   # Window / Button background
    base      = QColor("#1E1E1E")   # Input backgrounds
    mid       = QColor("#464649")   # Borders / separators
    text      = QColor("#DCDCDC")   # Primary text
    highlight = QColor("#2A82DA")   # Selection / accent

    dark_bg       = QColor("#252526")   # Alternate row background
    disabled_text = QColor("#808080")   # Greyed-out text
    white         = QColor("#FFFFFF")

    # ── Normal colour group ──────────────────────────────────────────────
    palette.setColor(QPalette.ColorRole.Window,          bg)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            base)
    palette.setColor(QPalette.ColorRole.AlternateBase,   dark_bg)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     base)
    palette.setColor(QPalette.ColorRole.ToolTipText,     text)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.Button,          bg)
    palette.setColor(QPalette.ColorRole.ButtonText,      text)
    palette.setColor(QPalette.ColorRole.BrightText,      white)
    palette.setColor(QPalette.ColorRole.Link,            highlight)
    palette.setColor(QPalette.ColorRole.Highlight,       highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, white)
    palette.setColor(QPalette.ColorRole.Mid,             mid)
    palette.setColor(QPalette.ColorRole.Dark,            QColor("#1A1A1C"))
    palette.setColor(QPalette.ColorRole.Shadow,          QColor("#141414"))
    palette.setColor(QPalette.ColorRole.Light,           QColor("#3F3F46"))

    # ── Disabled colour group ────────────────────────────────────────────
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled_text)

    return palette


# ─────────────────────────────────────────────────────────────────────────── #
#  Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    """
    Application entry point.

    Sets up QApplication with Fusion style, dark palette, and the global
    font, then launches MainWindow.
    """
    # High-DPI support (must be set before QApplication is constructed)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("FrameForge")
    app.setOrganizationName("AutonomousDriving")

    # Style + theme
    app.setStyle("Fusion")
    app.setPalette(build_dark_palette())

    # Global font — platform-specific typeface, fixed 9 pt size
    if platform.system() == "Windows":
        font = QFont("Segoe UI", 9)
    else:
        font = QFont("Ubuntu", 9)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
