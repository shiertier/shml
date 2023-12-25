import logging

from shml.caformer import build_caformer
from shml.ml_decoder import ml_decoder_b
from shml.tresnet import tresnet_f, tresnet

logger = logging.getLogger(__name__)

def create_model(model_name, num_classes, args, frelu=True):
    """Create a model
    """
    tres = tresnet_f if frelu else tresnet

    model_params = {'num_classes':num_classes}
    args.num_classes = num_classes

    if model_name == 'tresnet_m':
        model = tres.TResnetM(model_params)
    elif model_name == 'tresnet_d':
        model = tres.TResnetD(model_params)
    elif model_name == 'tresnet_l':
        model = tres.TResnetL(model_params)
    elif model_name == 'tresnet_xl':
        model = tres.TResnetXL(model_params)
    elif model_name == 'caformer_m36':
        model = build_caformer('caformer_m36_384', args)
    elif model_name == 'caformer_s36':
        model = build_caformer('caformer_s36_384', args)
    else:
        raise NotImplementedError("model: {} not found !!".format(model_name))

    if model_name.startswith('tresnet'):
        model = ml_decoder.add_ml_decoder_head(model, num_classes=num_classes, num_of_groups=args.num_of_groups,
                                               decoder_embedding=args.decoder_embedding)

    return model
