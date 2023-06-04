import glob
import cv2
import numpy as np

# формирует массив для сохранения в датасет
# pathNear - путь к папке с картинками цифр
def makeFromPics(pathNear):
    imgs = []
    lbls = []
    numberListQv = glob.glob(fr"./{pathNear}/*")
    for nameNum in range(len(numberListQv)):
        pathAll = numberListQv[nameNum].split('\\')
        notFullName = pathAll[-1]
        (name, jp) = notFullName.split('.')
        mean = int(name[-1])
        img = cv2.imread(fr"./{pathNear}/{notFullName}", cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)[1]
        k = np.array(img/255.0)
        if(nameNum % 10 == 0):
            print(nameNum)

        imgs.append(k)
        lbls.append(mean)
        # lbls.append([mean])

    testI = np.array(imgs)
    testL = np.array(lbls)
    return testI, testL

# Нарезает картинку на цифры
# placeStart - путь к картинкам с капчами
# placeFinish - путь, где будут цифры
def cutPicsInFolder(placeStart, placeFinish):
    piclist = glob.glob(fr'{placeStart}/*')
    for nameNum in range(len(piclist)):
        trashAndName = piclist[nameNum].split('\\')
        name = trashAndName[-1].replace('.png', '')
        print(name)
        littlePics = cutter(piclist[nameNum])
        i=0
        for [lp, zn] in littlePics:
            cv2.imwrite(f'{placeFinish}\\{i}-{name}-{lp}.png', zn)
            i+=1

# вырезает цифры с капчи и сохраняет в массив вместе с их значениями
# pic - картинка капчи, название вроде 12345.png
def cutter(pic):
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    binr = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)[1]
    trashAndPic = pic.split('\\')
    pic = trashAndPic[-1]
    ret = []
    x=30
    leng = 18
    for j in range(5):
        numberName = pic[j]
        picNum = binr[5:35, x+leng*j:x+leng*(j+1)]
        ret.append([numberName, picNum])
    return ret

# Более простая версия cutter, возвращает только список картинок-цифр, без значений
# pic - картинка капчи
def cutterSimple(pic):
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    # binarize the image
    binr = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)[1]
    ret = []
    x=30
    leng = 18
    for j in range(5):
        picNum = binr[5:35, x+leng*j:x+leng*(j+1)] / 255
        ret.append(picNum)
    return ret