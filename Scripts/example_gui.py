from test_gui import Ui_Dialog;
from PyQt5 import QtCore, QtGui, QtWidgets

def fun1(ui):
    print("New stuff"+str(ui.doubleSpinBox.value()))
    return 0



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()

    f1 = lambda : fun1(ui)   # for callbacks
    ui.pushButton.clicked.connect(f1)

    sys.exit(app.exec_())

