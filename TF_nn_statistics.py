import pickle

class ConvLayer():
    def __init__(self, aperture, zin=0, zout=0):
        self._aperture = aperture
        self._zin = zin
        self._zout = zout
    
    def __str__(self):
        return f'Aperture = {self._aperture} | input depth = {self._zin} | filters = {self._zout}'


def get_zin(layers : dict, inbound_nodes : dict) -> int:
    return 0

if __name__ == "__main__":
    fname = "configs/nn_struct_centernet.pkl"
    with open(fname, "rb") as f:
        struct = pickle.load(f)
    
    layers = struct['layers']

    layer_types = []
    conv_layers = []
    for l in layers:
        if l['class_name'] not in layer_types:
            layer_types.append(l['class_name'])
        if l['class_name'] == 'Conv2D':
            aperture = l['config']['kernel_size']
            zin = get_zin(layers, l['inbound_nodes'])
            zout = l['config']['filters']
            conv_layers.append(ConvLayer(aperture, zin, zout))

    print(layer_types)
    for l in conv_layers:
        print(l)