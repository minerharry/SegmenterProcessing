from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

class ColorButton(QtWidgets.QPushButton):
    '''
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    '''

    colorChanged = pyqtSignal(object)

    def __init__(self, *args, color=None, **kwargs):
        super(ColorButton, self).__init__(*args, **kwargs)

        self._selected_color = None
        self._default = color
        self.pressed.connect(self.onColorPicker)

        # Set the initial/default state.
        self.setColor(self._default)

    def setColor(self, color):
        if color != self._selected_color:
            self._selected_color = color
            self.colorChanged.emit(color)

        if self._selected_color:
            self.setStyleSheet("ColorButton {background-color: %s;}" % self._selected_color)
        else:
            self.setStyleSheet("")

    def color(self):
        return self._selected_color

    def onColorPicker(self):
        '''
        Show color-picker dialog to select color.

        Qt will use the native dialog by default.

        '''
        dlg = QtWidgets.QColorDialog(self)
        if self._selected_color:
            dlg.setCurrentColor(QtGui.QColor(self._selected_color))

        if dlg.exec():
            self.setColor(dlg.currentColor().name())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.setColor(self._default)
            print(self.color())

        return super(ColorButton, self).mousePressEvent(e)

class App(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.button = ColorButton(self)
        self.show()
        # self.layout().addWidget(self.button)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec())