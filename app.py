import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from tqdm import tqdm

# Load the machine learning model
mod = load_model('best_mod.h5')

# Define the list of Arabic characters for prediction
arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                    'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                    'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

# Create the main window and set its properties
root = tk.Tk()
root.title("Arabic Character Recognition")
root.geometry('1300x1000')
root.configure(bg='white')

# Create the header label
header = tk.Label(root, text="Arabic Character Recognition", font=('Arial', 28, 'bold'), bg='white')
header.pack(pady=30)

# Create the canvas to display the image
canvas = tk.Canvas(root, width=500, height=500, bg="white", highlightthickness=2, highlightbackground='black')
canvas.pack(pady=20)
'''pred_label = tk.Label(root, font=('Arial', 24, 'bold'), bg='white')
pred_label.pack(pady=10)'''
# Define function to open file dialog and load image
def open_image():
    global image
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    # Resize the image and calculate the center position
    max_size = (500, 500)
    image.thumbnail(max_size)
    # Create a new image with 10 times the size of the original image
    new_size = (image.width*10, image.height*10)
    zoomed_image = image.resize(new_size, Image.NEAREST)
    # Calculate the center position of the zoomed image
    x = (500 - zoomed_image.width) // 2
    y = (500 - zoomed_image.height) // 2
    # Create the PhotoImage and display it on the canvas
    photo = ImageTk.PhotoImage(zoomed_image)
    canvas.image = photo # keep a reference to the image to prevent garbage collection
    canvas.delete("all") # clear any existing image on the canvas
    canvas.create_image(x, y, anchor=tk.NW, image=photo)
    #pred_label.configure(text="", fg='#283747') # Clear previous prediction
    #pred_label.place(x=250, y=y-60)
# Define function to process the image
def process_image():
    global pred
    images = []
    img = image
    img = img.resize((32,32))
    images.append(np.array(img))
    img = images[0]/255
    img = img.reshape(-1,32,32,1)
    pred = np.argmax(mod.predict(img, verbose=0),axis = 1)[0]
   # pred_label.configure(text=arabic_characters[pred], font=('Arial', 24, 'bold'), fg='#283747')
    bbox = (canvas.winfo_width()//2-50, canvas.winfo_height()//2-50, canvas.winfo_width()//2+50, canvas.winfo_height()//2+50)
    pred_box = tk.Label(root, text=arabic_characters[pred], font=('Arial', 24, 'bold'), bg='white', fg='red')
    pred_box.place(x=bbox[0], y=bbox[1], width=bbox[2]-bbox[0], height=bbox[3]-bbox[1])
 
# Create the open button
open_button = tk.Button(root, text="Open Image", font=('Arial', 20), bg='#EB984E', fg='white', command=open_image)
open_button.pack(pady=10)

# Create the process button
process_button = tk.Button(root, text="Process Image", font=('Arial', 20), bg='#2E86C1', fg='white', command=process_image)
process_button.pack(pady=10)

# Create the prediction label

# Start the main loop
root.mainloop()
