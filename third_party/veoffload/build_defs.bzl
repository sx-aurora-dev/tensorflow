# -*- Python -*-

def if_ve(if_true, if_false = []):
    return select({
                   str(Label("//third_party/veoffload:using_ve")): if_true,
                   "//conditions:default": if_false
                   })
