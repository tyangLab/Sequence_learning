import base64
import pickle


def obj2base64(obj):
    """
    Public object2base64string func. Used in serized storage.
    :param obj:
    :return:
    """
    sobj = pickle.dumps(obj, protocol=2)  # support for python2
    bobj = base64.b64encode(sobj)
    return bobj.decode("utf-8")  # support for python2


def base642obj(bobj):
    sobj = base64.b64decode(bobj)
    obj = pickle.loads(sobj)
    return obj


def base642obj3(bobj):
    """
    decoder for py3
    """
    sobj = base64.b64decode(bobj)
    obj = pickle.loads(sobj, encoding='latin1')
    return obj