
def pose_acc(pred, target, hw, th=10):
    h = hw[0] if isinstance(hw[0], int) else int(hw[0][0]) 
    w = hw[1] if isinstance(hw[1], int) else int(hw[1][0])
    pred[:, 0::2] = pred[:, 0::2] * w
    pred[:, 1::2] = pred[:, 1::2] * h
    pred[pred < 0] = 0

    target[:, 0::2] = target[:, 0::2] * w
    target[:, 1::2] = target[:, 1::2] * h

    th = th
    acc = []
    for p, t in zip(pred, target):
        distans = ((t[0] - p[0])**2 + (t[1] - p[1])**2)**0.5
        if distans > th:
            acc.append(0)
        elif distans > 1:
            acc.append((th - distans) / (th - 1))
        else:
            acc.append(1)
    return sum(acc) / len(acc)


def audio_acc(pred, target):
    pred = pred[0] if len(pred.shape)==2 else pred # onnx shape(d,), tflite shape(1,d)
    pred = pred.argsort()[::-1][:5]
    correct = (target==pred).astype(float)
    acc = (correct[0], correct.max()) # (top1, top5) accuracy

    return acc