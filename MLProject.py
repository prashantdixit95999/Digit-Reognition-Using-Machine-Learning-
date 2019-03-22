#Handwriting Recognition,
#We will also use Image Processing


import matplotlib.pyplot as plt
from sklearn import datasets,svm
digits=datasets.load_digits()  #here it is not as basic dictionary it is a dataframe
print "digits :",digits.keys()
print "digits.target-----------",digits.target


images_and_labels=list(zip(digits.images,digits.target))
print "len(images_and_labels)",len(images_and_labels)

for index,[image,label] in enumerate(images_and_labels[:5]): #for black 0 and for white it is max
    print "index:",index,"image:\n",image, "  Label:",label             #Thus we are cutting in suh a way that we are getting data in three parts as array Index ,Image and Target
    plt.subplot(2,5,index+1) #Position numbering starts from 1
    plt.axis('on') #used to on the ticks
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest') #used to show the Image
    #cmap is Colour mapping, gray r is black and white   , interpolation is used to repreent every dot as rectamglw and is used to represent every fdot uniquely distinguishable
    #Used to show the exact boundary by interpolatiron
    plt.title('Training %i'%label)
    #Image is multidimensional matrix and the intensity is represented in it
#plt.show()

#to apply a classification on this data we need to flatten the image
#to turn the data in a (sample,feture) matrix:
n_samples=len(digits.images)
print "n_samples :",n_samples

imageData=digits.images.reshape((n_samples,-1)) #Here image was of 2 D and it is being converted to 1 D
print "After reshaped : len(imageData[0]):",len( imageData[0])

#create a Classifies : a support vector classifier
classifier=svm.SVC(gamma=0.001) #learning rate shpuld always be small and represented by gamma
#We learn the digits on the first half of the digits
classifier.fit(imageData[ :n_samples//2],digits.target[ :n_samples//2]) #Hypothesis space ke best hypothesis instance ko calculate karta fit
#Now predict the value o the digit on the second half
expected=digits.target[n_samples//2: ]
predicted=classifier.predict(imageData[n_samples//2:]) #here we will apply the Mahine Learning so we will use the Processed Data
#imagedata being calculated can't be printed on screen as it's data has been processed an d can be used in calculation
#But if we want to print the first row of image data then probably it will be used only in Machine Learning
#and the data is being obtained as digits.images
#but to fit we use the processed image data for training purpose

images_and_predictions=list(zip(digits.images[n_samples//2:],predicted))
for index,[image,prediction] in enumerate(images_and_predictions[:5]):
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction:%i' % prediction)
print 'Original Values :',digits.target[n_samples//2: (n_samples//2)+5]
plt.show()
classifier=svm.SVC(gamma=0.001)
classifier.fit(imageData[:],digits.target[:])
#used to process live image resize it to 8*8
from scipy.misc import imread,imresize,bytescale

img=imread("thr.jpeg") #Matrix Format read hota
img=imresize(img,(8,8)) #changes it to 8*8 images
img=img.astype(digits.images.dtype) #changes it to datatype needed i.e. Pandas
img=bytescale(img,high=16.0,low=0) #resolution change to 0 to 16 as same resolution
#in training of images
print " img :======\n",img.shape, "\n",img #Here we have changed it to 8*8 and total made of 3 colours so [8,8,3]
x_testData=[]
for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0)
print "x_testData :\n",x_testData

print "len(x_testData):\n",len(x_testData)
x_testData=[x_testData] #Converted to 2d Array and thus represents 1 amage and it's data
print "len(x_testData):\n",len(x_testData)

print "machine Output=",classifier.predict(x_testData)
plt.show()

#After whole loop , the loop runs 64 times thu the 8*8 image is converted to 1*64

#Basiclly what we are trying to do is that we want to detect the number given some
#Images in it
#Image Processing is most important in identifying the Pattern

#Benefits of This project - Can save life in case of Tsunami or any other calamities
#used in japan
#here basically we are trying to identify the digit only
#data which is being obtained ater processing the image is uses as X in Machine Learning
#More the better the training of Machine the output of Machine wil be that much Accurate
#Genrally we try to follow a  pattern in Machine learning
#Support Vecor Machine Algorithm (SVM) is used in image Processing to match Imagess
#1.Scalar Value-it represents  a single value
#2.Vector or 1-D Array
#3.Matrix or 2-d Array

#SVM algorithm try to Maximise the minimum distance between two different classes.
# In SVM we generslly try to find the best fit line which separates the boundary of to classes with minimum error
#There will be two lines separating them such that the minimu distance betwen two different classes will be maximised
#But to find this we need to plot the best line First on both circle and dot example
#And the Average of the two lines will be the Final Line
#Here to plot this we will use Classificaton as the output of digit can be any of 0 to 9
#Classification used Here
#SVM is known as Support Vector Machine as it contains the the row or column vector in ordr to support
#to design the Boundary Line
#Use of SVM Algorithm
#Sklearn already provides database of 2000 images
#We will use this for training purpose and to test it we will supply it ourself
#Support Vector Classifier is actual logic code which separatetd(SVC)

