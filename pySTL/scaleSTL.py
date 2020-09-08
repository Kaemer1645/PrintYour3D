import pySTL
def scaleSTL():
    model = pySTL.STLmodel(self.dlg.lineEdit2.text())  # --- podane z menu, musi byc wypelnione ---
    scale = float(self.dlg.lineEdit.text())
    model.scale(scale)
    model.write_text_stl(self.dlg.lineEdit.text()+str(scale^(-1)))   # --- miejsce zapisu

