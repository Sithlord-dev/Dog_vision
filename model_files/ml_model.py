# Data analysis libraries
import numpy as np

## keras model layers and callback:
from keras.models import Sequential
from keras.layers import InputLayer, Lambda

## Preprocessing:
from keras.preprocessing.image import load_img, img_to_array


# Functions

def images_to_array(source, target_size=(331, 331, 3)):
    images = np.zeros([1, target_size[0], target_size[1], target_size[2]], dtype=np.uint8)
    img = load_img(source, target_size=target_size)
    images[0] = img_to_array(img)
    del img
    return images


def get_features(model, preprocess_input, images, target_size=(331, 331, 3)):
    conv_base = model(input_shape=target_size,
                      include_top=False,
                      weights='imagenet',
                      pooling='avg')

    cnn = Sequential([
        InputLayer(input_shape=target_size),  # input layer
        Lambda(preprocess_input),  # preprocessing layer
        conv_base])  # base model

    features = cnn.predict(images)

    print('feature-map shape: {}'.format(features.shape))
    return features


def make_prediction(model, source, target_size=(331, 331, 3)):
    images = images_to_array(source, target_size=target_size)

    ## run feature extraction
    # Resnet50
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
    resnet_preprocess = preprocess_input
    resnet_features = get_features(InceptionResNetV2, resnet_preprocess, images)
    # Xception
    from keras.applications.xception import Xception, preprocess_input
    xception_preprocess = preprocess_input
    xception_features = get_features(Xception, xception_preprocess, images)
    # InceptionV3
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    inception_preprocess = preprocess_input
    inception_features = get_features(InceptionV3, inception_preprocess, images)
    # Nasnet
    from keras.applications.nasnet import NASNetLarge, preprocess_input
    nasnet_preprocessor = preprocess_input
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, images)

    final_features = np.concatenate([resnet_features, xception_features, inception_features, nasnet_features], axis=1)

    del resnet_features, xception_features, inception_features, nasnet_features

    # make predictions
    predictions = model.predict(final_features)

    return predictions


def get_labels(predictions):
    labels_dict = {0: 'A Boston bull',
                   1: 'A Dingo',
                   2: 'A Pekinese',
                   3: 'A Bluetick',
                   4: 'A Golden retriever',
                   5: 'A Bedlington terrier',
                   6: 'A Borzoi',
                   7: 'A Basenji',
                   8: 'A Scottish deerhound',
                   9: 'A Shetland sheepdog',
                   10: 'A Walker hound',
                   11: 'A Maltese dog',
                   12: 'A Norfolk terrier',
                   13: 'An African hunting dog',
                   14: 'A Wire-haired fox terrier',
                   15: 'A Redbone',
                   16: 'A Lakeland terrier',
                   17: 'A Boxer',
                   18: 'A Doberman',
                   19: 'An Otterhound',
                   20: 'A Standard schnauzer',
                   21: 'An Irish water spaniel',
                   22: 'A Black-and-tan coonhound',
                   23: 'A Cairn',
                   24: 'An Affenpinscher',
                   25: 'A Labrador retriever',
                   26: 'An Ibizan hound',
                   27: 'An English setter',
                   28: 'A Weimaraner',
                   29: 'A Giant schnauzer',
                   30: 'A Groenendael',
                   31: 'A Dhole',
                   32: 'A Toy poodle',
                   33: 'A Border terrier',
                   34: 'A Tibetan terrier',
                   35: 'A Norwegian elkhound',
                   36: 'A Shih-tzu',
                   37: 'An Irish terrier',
                   38: 'A Kuvasz',
                   39: 'A German shepherd',
                   40: 'A Greater swiss mountain dog',
                   41: 'A Basset',
                   42: 'An Australian terrier',
                   43: 'A Schipperke',
                   44: 'A Rhodesian ridgeback',
                   45: 'An Irish setter',
                   46: 'An Appenzeller',
                   47: 'A Bloodhound',
                   48: 'A Samoyed',
                   49: 'A Miniature schnauzer',
                   50: 'A Brittany spaniel',
                   51: 'A Kelpie',
                   52: 'A Papillon',
                   53: 'A Border collie',
                   54: 'An Entlebucher',
                   55: 'A Collie',
                   56: 'A Malamute',
                   57: 'A Welsh springer spaniel',
                   58: 'A Chihuahua',
                   59: 'A Saluki',
                   60: 'A Pug',
                   61: 'A Malinois',
                   62: 'A Komondor',
                   63: 'An Airedale',
                   64: 'A Leonberg',
                   65: 'A Mexican hairless',
                   66: 'A Bull mastiff',
                   67: 'A Bernese mountain dog',
                   68: 'An American staffordshire terrier',
                   69: 'A Lhasa',
                   70: 'A Cardigan',
                   71: 'An Italian greyhound',
                   72: 'A Clumber',
                   73: 'A Scotch terrier',
                   74: 'An Afghan hound',
                   75: 'An Old english sheepdog',
                   76: 'A Saint bernard',
                   77: 'A Miniature pinscher',
                   78: 'An Eskimo dog',
                   79: 'An Irish wolfhound',
                   80: 'A Brabancon griffon',
                   81: 'A Toy terrier',
                   82: 'A Chow',
                   83: 'A Flat-coated retriever',
                   84: 'A Norwich terrier',
                   85: 'A Soft-coated wheaten terrier',
                   86: 'A Staffordshire bullterrier',
                   87: 'An English foxhound',
                   88: 'A Gordon setter',
                   89: 'A Siberian husky',
                   90: 'A Newfoundland',
                   91: 'A Briard',
                   92: 'A Chesapeake bay retriever',
                   93: 'A Dandie dinmont',
                   94: 'A Great pyrenees',
                   95: 'A Beagle',
                   96: 'A Vizsla',
                   97: 'A West highland white terrier',
                   98: 'A Kerry blue terrier',
                   99: 'A Whippet',
                   100: 'A Sealyham terrier',
                   101: 'A Standard poodle',
                   102: 'A Keeshond',
                   103: 'A Japanese spaniel',
                   104: 'A Miniature poodle',
                   105: 'A Pomeranian',
                   106: 'A Curly-coated retriever',
                   107: 'A Yorkshire terrier',
                   108: 'A Pembroke',
                   109: 'A Great dane',
                   110: 'A Blenheim spaniel',
                   111: 'A Silky terrier',
                   112: 'A Sussex spaniel',
                   113: 'A German short-haired pointer',
                   114: 'A French bulldog',
                   115: 'A Bouvier des flandres',
                   116: 'A Tibetan mastiff',
                   117: 'An English springer',
                   118: 'A Cocker spaniel',
                   119: 'A Rottweiler'}
    breed = labels_dict[np.argmax(predictions)]
    return breed
