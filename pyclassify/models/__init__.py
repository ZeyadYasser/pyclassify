from pyclassify.models.squeezenet import SqueezeNet

models_dict = {
    'squeeze_net': SqueezeNet,
}

def get_backend(model_name, classes):
    if model_name not in models_dict:
        raise Exception('Model: "{0}" does not'
                        'exist.'.format(model_name))
    return models_dict[model_name](classes)