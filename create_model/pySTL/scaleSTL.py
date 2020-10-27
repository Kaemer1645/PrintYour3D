#this code is designed to create scale of existed model .stl file


from .pySTL import STLmodel

class Scalator:

    def __init__(self, scale_text, cmbScale):
        self.cmbScale = cmbScale
        self.scale_text = scale_text
        self.small_scale = None
        self.big_scale = None
    def scaleSTL(self):
        model = STLmodel(str(self.scale_text.text()))
        scale = str(self.cmbScale.currentText())
        scale = scale.split(':')
        if scale[0] == '1':
            self.small_scale = int(scale[1])
            print(self.small_scale)
            model.scale(float(1/self.small_scale))
            model.write_text_stl(str(self.scale_text.text()).strip('.stl') +'_' + str(self.small_scale) + '.stl')
        else:
            self.big_scale = int(scale[0])
            print(self.big_scale)
            model.scale(float(self.big_scale))
            model.write_text_stl(str(self.scale_text.text()).strip('.stl')+ '_' + str(self.big_scale) + '.stl')

