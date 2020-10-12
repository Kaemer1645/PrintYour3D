from .pySTL import STLmodel

class Scalator:

    def __init__(self, dlg):
        self.dlg = dlg
        self.small_scale = None
        self.big_scale = None
    def scaleSTL(self):
        model = STLmodel(str(self.dlg.lineEdit.text()))  # --- podane z menu, musi byc wypelnione ---
        scale = str(self.dlg.comboBox_4.currentText())
        scale = scale.split(':')
        if scale[0] == '1':
            self.small_scale = int(scale[1])
            print(self.small_scale)
            model.scale(float(1/self.small_scale))
            model.write_text_stl(str(self.dlg.lineEdit.text()).strip('.stl') +'_' + str(self.small_scale) + '.stl')
        else:
            self.big_scale = int(scale[0])
            print(self.big_scale)
            model.scale(float(self.big_scale))
            model.write_text_stl(str(self.dlg.lineEdit.text()).strip('.stl')+ '_' + str(self.big_scale) + '.stl')
        #model.scale(scale)
        #model.write_text_stl(str(self.dlg.lineEdit.text())+str(scale^(-1)))   # --- miejsce zapisu

    def kuzwar(self):
        return print('kukurydza')