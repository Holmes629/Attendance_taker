import tkinter as tk
from tkinter import filedialog
import cv2
import datetime
from PIL import ImageTk, Image
import face_recognition
import cv2
import os
import numpy as np
import pickle
from datetime import date 
from datetime import date,datetime
import pandas as pd


####################################### Page-1 (Loading Window) ##########################################
def page1():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')

    label1= tk.Label(root, text= "Attendance Taker", fg='#5755F3',font=('Arial',72))
    label2= tk.Label(root, text= "via Face Recognition", fg='#5755F3', font=("Aria", 18))
    label3= tk.Label(root, text= "By:\nAgrima\nBhanu Sai Akhil\nHitesh\nVaibhav", fg='#A155F3' ,font=("Arial", 18), justify="left")

    label1.grid(row=0, column=1, padx=400,pady=(250,10), columnspan=3)
    label2.grid(row=1, column=1, columnspan=3)
    label3.grid(row=2, column=3, columnspan=3, pady=(150,10))
    def des():
        root.destroy()
    root.after(2000,des)
    root.mainloop()

####################################### Page-2 (Home Window) ##########################################
def page2():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')

    def Train(): 
        root.destroy()
        page4()
    def Mark_Attendance():
        root.destroy()
        page3()
    def Check_Attendance():
        root.destroy()
        page5()

    button1= tk.Button(root, text='Mark Attendance', command=Mark_Attendance,font=('Arial',18), padx=10, pady=10, bg='#CA9DA7')
    button2= tk.Button(root, text='Train', command= Train,font=('Arial',18), padx=70, pady=10, bg='#CA9DA7')
    button3= tk.Button(root, text='Check Attendance', command=Check_Attendance,font=('Arial',18), pady=10, bg='#CA9DA7')

    button1.grid(row=0, column=1, columnspan=3, padx=700, pady= (160,60))
    button2.grid(row=1, column=1, columnspan=3, pady=60)
    button3.grid(row=2, column=1, columnspan=3, pady=(60,100))
    root.mainloop()
    

####################################### Page-3 (Attendance Taking Window) ##########################################
def page3():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')

    def Home():
        file=os.path.dirname(__file__)+"/data_set/attendance.csv"
        today = date.today()
        today = str(today.strftime("%d/%m/%Y"))
        
        if os.path.exists(file)==False:
            attendance={'Names':[],today:[]}
            for face in known_face_names:
                attendance['Names']+=[face]
                if face in attended and face!=None:
                    attendance[today]+=[1]
                elif face not in attended:
                    attendance[today]+=[0]
            df = pd.DataFrame(attendance, columns= ['Names', today])
            df.to_csv (file, index = False, header=True)
        else:
            df = pd.read_csv(file)
            if today not in df.columns:
                today_att=[]
                for face in known_face_names:
                    if face in attended:
                        today_att.append(1)
                    else:
                        today_att.append(0)
                new_column = pd.DataFrame({today:today_att})
                df = df.merge(new_column, left_index = True, right_index = True)
                df.to_csv(file, index = False)
            else:
                n=0
                df=pd.read_csv(file)
                today_att=df[today]
                for i in range(len(today_att)):
                    if today_att[i]==0:
                        face=known_face_names[i]
                        if face in attended:
                            today_att[n]=1
                        n+=1      
                    else:
                        n+=1
                new_column = pd.DataFrame({today:today_att})
                df = df.replace(df[today],new_column)
                df.to_csv(file, index = False)

        root.destroy()
        video.release()
        cv2.destroyAllWindows()
        page2()
    

    video= cv2.VideoCapture(0)
    img= video.read()[1]

    image_label= tk.Label(root, height=600, width=1100)
    Label2= tk.Label(root, text='Face is Not Recognized', height=5, width=90, bg='#E2BDC5',font=('Arial',18))
    button1= tk.Button(root, text='Home', command= Home,font=('Arial',18), padx=30, bg='#CA9DA7')
    
    image_label.grid(row=0, column=1, columnspan=3, padx=(10,20), pady=(10,10))
    Label2.grid(row=1, column=2, columnspan=2, padx=(10,20), pady=(10,10))
    button1.grid(row=0, column=4)
    i=0

    known_face_encodings = []
    known_face_names = []

    dataset=os.path.dirname(__file__)+'/dataset_faces.dat'
    with open(dataset, 'rb') as data:
        all_face_encodings = pickle.load(data)
    
    known_face_encodings=np.array(list(all_face_encodings.values()))
    known_face_names=list(all_face_encodings.keys())
    known_face_names=sorted(known_face_names)
    attended=[]

    while True:
        img= video.read()[1]
        img=cv2.flip(img,1)

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        if process_this_frame:
            rgb_small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                hour = int(datetime.now().hour)
                mm = int(datetime.now().minute)
                Label2['text']= 'Hello '+name+ ', marked your attendance at '+ str(hour)+':'+str(mm)
                if name=='Unknown':
                    Label2['text']= name+ ' face is recognized no attendace is marked, \n please train the data set with your image to recognize you next time'
                face_names.append(name)
                if name not in attended and name!='' and name!='Unknown':
                    attended.append(name)
                
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            img=cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            img=cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            if name=='Unknown' or name=='':
                img=cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                img=cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            img=cv2.putText(img, name, (left + 10, bottom -6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img, (0, 0), fx=1, fy=1)
        img= ImageTk.PhotoImage(Image.fromarray(img))
        image_label['image']= img
        

        root.update()



####################################### Page-4 (Training Options Window) ##########################################
def page4():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')

    def Home():
        root.destroy()
        page2()
    def Del_data():
        root.destroy()
        page6()
    def Train_cam():
        root.destroy()
        page7()
    def openFile():
        root.destroy()
        page8()

    button1= tk.Button(root, text='Delete Data', command= Del_data,font=('Arial',18), padx=40, pady=10, bg='#CA9DA7')
    button2= tk.Button(root, text='Train via Camera',  command= Train_cam, font=('Arial',18), padx=10, pady=10, bg='#CA9DA7')
    button3= tk.Button(root, text='Upload Photo',command=openFile, font=('Arial',18), padx=30,pady=10, bg='#CA9DA7')
    button4= tk.Button(root, text='Home', command= Home,font=('Arial',18), padx=30, bg='#CA9DA7')

    button1.grid(row=0, column=1, columnspan=3, padx=700, pady= (160,60))
    button2.grid(row=1, column=1, columnspan=3, pady=60)
    button3.grid(row=2, column=1, columnspan=3, pady=(60,70))
    button4.grid(row=3, column=3, columnspan=3, pady=(60,10))

    root.mainloop()

####################################### Page-5 (Checking Attendance Window) ##########################################
def page5():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')
   
    def Home():
        root.destroy()
        page2()

    # get the list of files
    path= os.path.dirname(__file__)+'/data_set'
    flist = os.listdir(path)
    
    lbox = tk.Listbox(root,height=20, width=70, font=('Arial',20))
    button4= tk.Button(root, text='Home', command= Home,font=('Arial',18), padx=30, bg='#CA9DA7')

    lbox.grid(row=0, column=0, columnspan=3, pady=(20,10),padx= (20,10))
    button4.grid(row=0, column=3, columnspan=3, pady=(20,10),padx=(100,10))
    
    # THE ITEMS INSERTED WITH A LOOP
    for item in flist:
        lbox.insert(tk.END,item)
    
    
    def opensystem(event):
        x = lbox.curselection()[0]
        file= path+ '\\'+lbox.get(x)
        os.startfile(file)
        
    lbox.bind("<Double-Button-1>", opensystem)
    
    root.mainloop()


####################################### Page-6 (Deleting Data Window) ##########################################
def page6():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')


    def Home():
        root.destroy()
        page2()
    def dlt():
        n= lbox.get(lbox.curselection()[0])
        button3= tk.Button(root, font=('Arial',16),bg='#F35E8B')
        with open(dataset, 'rb') as data:                         # opening the dataset.dat file
            all_face_encodings = pickle.load(data)
            lbox.delete(0,len(all_face_encodings)+1)
            del all_face_encodings[n]
            if len(list(all_face_encodings.keys()))==0:             # if no user data present return error
                button3['text']= 'NO USER DATA IS PRESENT'
            else:
                button3['text']= 'SELECTED USER DATA IS DELETED'
                for i in list(all_face_encodings.keys()):
                    if i!='':
                        lbox.insert(tk.END,i)
            with open(dataset, 'wb') as data:                       # write the new data( with removed facial data of requested user to file)
                pickle.dump(all_face_encodings, data)
            data.close()
        button2['state']= 'disabled'
        button3.grid(row=2, column=0, columnspan=3,pady=(100,10),padx=(0,0))

    lbox= tk.Listbox(root, font=('Arial',18),width=80)
    button4= tk.Button(root, text='Home', command= Home,font=('Arial',18), padx=30, bg='#CA9DA7')
    dataset=os.path.dirname(__file__)+'/dataset_faces.dat'
    with open(dataset, 'rb') as data:                         # opening the dataset.dat file
        all_face_encodings = pickle.load(data)
        if len(list(all_face_encodings.keys()))==0:             # if no user data present return error
            print('\n NO USER DATA IS PRESENT')
        else:
            for i in list(all_face_encodings.keys()):
                if i!='':
                    lbox.insert(tk.END,i)                # list all the names in the dataset.dat file here "keys=names"

    canvas1 = tk.Canvas(root, width=400, height=300)

    button1 = tk.Button(text='Please double click on the data you want to delete ', font=('Arial',18),bg='#8E5EF3')
    canvas1.create_window(200, 180, window=button1)
    button2 = tk.Button(text='DELETE', command= dlt,font=('Arial',18),state='disabled')

    button1.grid(row=1, column=0, columnspan=3,pady=(100,10),padx=(0,0))
    button2.grid(row=1, column=1, columnspan=3,pady=(100,10),padx=(500,0))
    
    
    lbox.grid(row=0, column=0, columnspan=3,pady=(50,50),padx=(50,50))
    button4.grid(row=0, column=3, columnspan=3, pady=(60,10))
    def opensystem(event):
        button2['state']='active'


    lbox.bind("<Double-Button-1>", opensystem)
    root.mainloop()

####################################### Page-7 (Training Via Camera Window) ##########################################
def page7():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')
    name=['']
    def Home():
        root.destroy()
        page2()

    def submit():
        name[0]= entry1.get()
        if name[0]!='':
            root.destroy()

    canvas1 = tk.Canvas(root, width=400, height=300)
    entry1 = tk.Entry(root,font=('Arial',18)) 
    canvas1.create_window(200, 140, window=entry1)

    button1 = tk.Button(text='Please input the name of the data: ',font=('Arial',18),bg='#8E5EF3')
    button2 = tk.Button(text='Submit', command=submit ,font=('Arial',18),bg='#CA9DA7')
    canvas1.create_window(200, 180, window=button1)
    button3 = tk.Button(text='If no image is detected or recognised, you will be redirected to this page again, \n else it will be directed to Home page ',font=('Arial',18),bg='#5EF396')
    button4= tk.Button(root, text='Home', command= Home,font=('Arial',18), padx=30, bg='#CA9DA7')

    button1.grid(row=0, column=0, columnspan=3,pady=(200,10),padx=(100,0))
    entry1.grid(row=0, column=2, columnspan=3, pady=(200,10),padx=(300,0))
    button2.grid(row=1, column=2, columnspan=3,padx=(300,0))
    button3.grid(row=3, column=0, columnspan=3,pady=(200,10),padx=(200,0))
    button4.grid(row=3, column=3, columnspan=3, pady=(200,10),padx=(100,0))
    
    root.mainloop()


    name= name[0]              
    #print('Please look at the camera for 5 seconds :)') # photo for face encoding
    if name!='':
        path=os.path.dirname(__file__)+'\Photos'            #path as Photos in our dir; If not present create a directory 
        if os.path.exists(path)==False:
            os.mkdir(path)

        cam= cv2.VideoCapture(0)                            #capturing the image

        start_time = datetime.now()
        diff = 5 - (datetime.now() - start_time).seconds # converting into seconds
        while( diff > 0 ):
            result, image=cam.read()
            image=cv2.flip(image,1)
            cv2.putText(image, str(diff), (100,100), cv2.FONT_HERSHEY_SIMPLEX , 3, (0, 0, 255), 3, cv2.LINE_AA) 
            cv2.imshow('image',image)
            diff =  5 - (datetime.now() - start_time).seconds
            if cv2.waitKey(1) & 0xFF == ord('q'):           # exit if typed alphabet q
                break
    
        result, image=cam.read()
        cam.release()
        cv2.destroyAllWindows() 
        
        if result:                                          #if result =1 then store the image to pgotos folder with their name.jpg
            cv2.imwrite(os.path.join(path,name+'.jpg'),image) 
            print('Sucessfully added '+name+' to the database')
            
            image=os.path.join(path,name+'.jpg')            # image path for software to read and encode the facial data
            dataset=os.path.dirname(__file__)+'/dataset_faces.dat'  # creating a file 'dataset_faes.dat' for storing the facial data

            if os.path.exists(dataset)==False:              # if dataset file is not avialble then create dataset
                data=open(dataset,'w')
                data.close()

            if os.stat(dataset).st_size==0:                 # if the dataset file is empty then append {} to the data as python can't read a empty file
                with open(dataset,'wb') as data:
                    pickle.dump({},data)

            with open(dataset, 'rb') as data:               #opening dataset file to read the data
                all_face_encodings = pickle.load(data)      # loading the previoud data to append the new face data

            img=face_recognition.load_image_file(image)     # loading the image by using module which mload an image RGB values to numpy array
            if face_recognition.face_encodings(img)!=[]:
                all_face_encodings[name] = face_recognition.face_encodings(img)[0]  # convert image to HOG and create a 128 dataset values for facial recognition and adding it to dictionary with their name
            else:
                print('No image detected, Please try again')
                page7()
            with open(dataset, 'wb') as data:               # opening dataset.dat to write new facial data
                pickle.dump(all_face_encodings, data)
            data.close()                                    # closing the file to avoid errors
        else:
            print('No image detected, Please try again')        #if no image detected return this error
        page1()
        page2()

 

####################################### Page-8 (Uploading Window) ##########################################
def page8():
    root= tk.Tk()
    root.title('Attendance Taker')
    root.geometry('1600x1000')
    name=['']
    def Home():
        root.destroy()
        page2()

    def submit():
        name[0]= entry1.get()
        filepath = filedialog.askopenfilename(filetypes=[('Photo', ('*.jpg','*.jpeg','*.png'))])
        image=cv2.imread(filepath)
        if filepath!='':     
            path=os.path.dirname(__file__)+'\Photos'                                     #if result =1 then store the image to pgotos folder with their name.jpg
            cv2.imwrite(os.path.join(path,name[0]+'.jpg'),image) 
            print('Sucessfully added '+name[0]+' to the database')
            
            image=filepath            # image path for software to read and encode the facial data
            dataset=os.path.dirname(__file__)+'/dataset_faces.dat'  # creating a file 'dataset_faes.dat' for storing the facial data

            if os.path.exists(dataset)==False:              # if dataset file is not avialble then create dataset
                data=open(dataset,'w')
                data.close()

            if os.stat(dataset).st_size==0:                 # if the dataset file is empty then append {} to the data as python can't read a empty file
                with open(dataset,'wb') as data:
                    pickle.dump({},data)

            with open(dataset, 'rb') as data:               #opening dataset file to read the data
                all_face_encodings = pickle.load(data)      # loading the previoud data to append the new face data

            img=face_recognition.load_image_file(image)     # loading the image by using module which mload an image RGB values to numpy array
            if face_recognition.face_encodings(img)!=[]:
                all_face_encodings[name[0]] = face_recognition.face_encodings(img)[0]  # convert image to HOG and create a 128 dataset values for facial recognition and adding it to dictionary with their name
            else:
                print('No image detected, Please try again')
                page7()
            with open(dataset, 'wb') as data:               # opening dataset.dat to write new facial data
                pickle.dump(all_face_encodings, data)
            data.close()                                    # closing the file to avoid errors
            root.destroy()
            page9()
        else:
            print('No image detected, Please try again')        #if no image detected return this error
        if name[0]!='':
            root.destroy()

    canvas1 = tk.Canvas(root, width=400, height=300)
    entry1 = tk.Entry(root,font=('Arial',18)) 
    canvas1.create_window(200, 140, window=entry1)

    button1 = tk.Button(text='Please input the name of the data: ',font=('Arial',18),bg='#8E5EF3')
    button2 = tk.Button(text='Submit', command=submit ,font=('Arial',18),bg='#CA9DA7')
    canvas1.create_window(200, 180, window=button1)
    button3 = tk.Button(text='If no image is detected or recognised, you will be redirected to this page again, \n else it will be directed to Home page ',font=('Arial',18),bg='#5EF396')
    button4= tk.Button(root, text='Home', command= Home,font=('Arial',18), padx=30, bg='#CA9DA7')

    button1.grid(row=0, column=0, columnspan=3,pady=(200,10),padx=(100,0))
    entry1.grid(row=0, column=2, columnspan=3, pady=(200,10),padx=(300,0))
    button2.grid(row=1, column=2, columnspan=3,padx=(300,0))
    button3.grid(row=4, column=0, columnspan=3,pady=(200,10),padx=(200,0))
    button4.grid(row=3, column=3, columnspan=3, pady=(100,10),padx=(100,0))

    root.mainloop()

####################################### Page-9 (Upload success window) ##########################################
def page9():
    root = tk.Tk()
    #Set the geometry of Tkinter frame
    root.geometry("350x150")

    def ok():
        root.destroy()
        page2()
        
    tk.Label(root, text=" User uploaded succesfully", font=('Helvetica 14 bold')).pack(pady=20)
    tk.Button(root, text= "Ok", command= ok).pack()
    root.mainloop()          
page1()
page2()
# page3()
# page4()
# page5()
# page6()
# page7()
# page8()
# page9()



