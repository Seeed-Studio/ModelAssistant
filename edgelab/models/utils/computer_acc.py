
def pose_acc(pred, target, hw, th=10):
    h = hw[0]
    w = hw[1]

    pred = pred[0] if len(pred.shape)==2 else pred # onnx shape(d,), tflite shape(1,d)
    pred[0] = pred[0] * w
    pred[1] = pred[1] * h
    pred[pred < 0] = 0

    target[0] = target[0] * w
    target[1] = target[1] * h

    th = th
    acc = []
    p, t = zip(pred, target)
    distans = ((p[0] - p[1])**2 + (t[0] - t[1])**2)**0.5
    if distans > th:
        acc.append(0)
    elif distans > 1:
        acc.append((th - distans) / (th - 1))
    else:
        acc.append(1)
    return sum(acc) / len(acc)


def audio_acc(pred, target):
    import numpy as np
    pred = pred[0] if len(pred.shape)==2 else pred # onnx shape(d,), tflite shape(1,d)
    pred = pred.argsort()[::-1][:5]
    correct = (target==pred).astype(float)
    acc = (correct[0], correct.max()) # (top1, top5) accuracy

    return acc