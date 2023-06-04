import makeDataset as mdst
import pickle
from settings import *
from useModel import *

# нарезка капч на цифры, из первой папки во вторую
mdst.cutPicsInFolder(captchasForTest, numbersForTest)
mdst.cutPicsInFolder(captchasForTrain, numbersForTrain)

# обработка картинок с цифрами и преобразование в единый массив
(imagesTest, labelsTest) = mdst.makeFromPics(numbersForTest)
(imagesTrain, labelsTrain) = mdst.makeFromPics(numbersForTrain)

# сохранение результата
with open(pickleDataset, 'wb') as output:
    pickle.dump((imagesTest, labelsTest, imagesTrain, labelsTrain), output)

# обучение модели на получившемся датасете
teachModel(pickleDataset, digitsModelPath)

# использование модели
print("Результат распознавания картинки: " + picToCaptcha(pathToPic))
