import time
from fvcore.nn import FlopCountAnalysis, flop_count_table


def flop_count(model, input_dict):

    img = input_dict['img']
    img = img.unsqueeze(0)
    print("The shape of the input image is {}".format(img.shape))

    model = model.cuda()
    img = img.cuda()

    model.eval()
    flops = FlopCountAnalysis(model, img)
    print(flop_count_table(flops))

    t0 = 0
    for i in range(1200):
        model(img)
        if i == 200:
            t0 = time.time()
    t1 = time.time()
    fps = 1000 / (t1 - t0)
    print("The FPS of the model is {}".format(fps))
    return
