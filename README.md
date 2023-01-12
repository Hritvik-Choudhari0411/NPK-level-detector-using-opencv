# Process Diagnostic Tool

## Technolgy Stack / Tools

<p align="left">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original-wordmark.svg" alt="python" width="70" height="70" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original-wordmark.svg" alt="opencv" width="70" height="70"/>

</p>

## Installation

Download and install the required libraries for the python file by running the following commands in the terminal.

```
pip install numpy
pip install sklearn
pip install pandas
pip install os
pip install matplotlib
pip install pytesseract
```

## Extracting frames from real-time video input (In code we've a short clip instead):
```sh
video_path= r'C:\Users\hritv\Desktop\AKXA Tech\ZACL __ NPKpredictNEW (1).mp4'
cap= cv2.VideoCapture(video_path)
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%30 == 0:
        cv2.imwrite(r'C:\Users\hritv\Desktop\AKXA Tech\Frames\frame%d.jpg'%i,frame)
    i+=1

cap.release()
cv2.destroyAllWindows()
```
## Function to extract coordinates of ROI from an image frame:
```sh
scale= 0.5
circles=[]
counter=0
counter2=0
point1= []
point2= []
myPoints= []
myColor= []

def mousePoints(event,x,y,flags,params):
    global counter, point1, point2, counter2, circles, myColor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            point1 = int(x//scale), int(y//scale);
            counter+=1
            myColor = (random.randint(0,2)*200,random.randint(0,2)*200,random.randint(0,2)*200)
        elif counter == 1:
            point2 = int(x//scale), int(y//scale);
            Type= input('Enter Type')
            name= input('Enter name')
            myPoints.append([point1,point2,Type,name])
            counter= 0
        circles.append([x,y,myColor])
        counter2+=1
img1= cv2.imread(r'C:\Users\hritv\Desktop\AKXA Tech\Frames\frame30.jpg')
img1= cv2.resize(img1,(0,0),None,scale,scale)

while True:
    for x,y,color in circles:
        cv2.circle(img1,(x,y),3,color,cv2.FILLED)
    cv2.imshow('org img',img1)
    cv2.setMouseCallback('org img', mousePoints)
    if cv2.waitKey(1) & 0xFF == 27:
        print(myPoints)
        break
```

## Processing the ROI image and applying OCR to convert image to string and saving it in a dataframe:
```sh
frames= sorted(os.listdir(r'C:\Users\hritv\Desktop\AKXA Tech\Frames'),key=lambda x: int(x[5:][:-4]))
N=0
P=0
K=0
Data= pd.DataFrame(columns= ['N %','P %','K %'])
for i in frames:    
    roi= [[(760, 252), (836, 278), 'text', 'N %'], 
          [(764, 310), (832, 334), 'text', 'P %'], 
          [(762, 364), (838, 390), 'text', 'K %']]

    img= cv2.imread(r'C:\Users\hritv\Desktop\AKXA Tech\Frames\{}'.format(i))
    imgShow= img.copy()
    imgMask= np.zeros_like(imgShow)

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0],r[0][1]), (r[1][0],r[1][1]), (0,255,0), cv2.FILLED)
        imgShow= cv2.addWeighted(imgShow, 0.99, imgMask, 0.1,0)
        imgCrop= img[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if x == 0:
            gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            dist = cv2.distanceTransform(thresh1, cv2.DIST_L2, 5)
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            dist = (dist * 255).astype("uint8")
            thresh2 = cv2.threshold(dist, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            N= re.split('\n| ',pytesseract.image_to_string(thresh2, config= '--psm 7'))[0]
        elif x== 1:
            gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            dist = cv2.distanceTransform(thresh1, cv2.DIST_L2, 5)
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            dist = (dist * 255).astype("uint8")
            thresh2 = cv2.threshold(dist, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            P= re.split('\n| ',pytesseract.image_to_string(thresh2, config= '--psm 7'))[0]
        else:
            K= re.split('\n',pytesseract.image_to_string(imgCrop))[0]
            Data= Data.append({'N %':N,'P %':P,'K %':K}, ignore_index= True)
    if (float(N) < 0.95*12 or float(N) > 1.05*12) or (float(P) < 0.95*24.5 or float(P) > 1.05*24.5) or (float(K) < 0.95*5 or float(K) > 1.05*5):
        print('STOP!, Threshold value crossed.')
        break
```
## Plots:
