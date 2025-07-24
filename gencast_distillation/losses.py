LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

def get_loss(name):
    return LOSS_REGISTRY[name]

@register_loss("denoiser_l2")
def denoiser_l2_loss(student_pred, teacher_pred, **kwargs):
    return ((student_pred - teacher_pred) ** 2).mean()