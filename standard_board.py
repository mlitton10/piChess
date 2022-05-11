import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

##------------------------------------------------------------------------------------
# Read the image files in

img_dir = 'test_images/'
img_file = 'blank_board_2.jpeg'

img_loc = ''.join([img_dir,img_file])
img = cv.imread(img_loc)
img2 = cv.imread(img_loc)
img3 = cv.imread(img_loc)
img4 = cv.imread(img_loc)
##------------------------------------------------------------------------------------

# Image processing

# Create gray scale image and vectorize it
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.uint8(gray)

img2_arr = np.uint8(img2) # turn image into unsigned 8 bit integer array
print(img2_arr.shape)

# find edges with the Canny edge detector algo
edges = cv.Canny(gray,100,200)

# edges are sloppy and don't extend beyond real edge, create 'clean' line that extends to image border
find_edges_0 = np.mean(edges,axis=0)  # find the average value of pixel in each column
find_edges_1 = np.mean(edges,axis=1)  # find avearge value of pixel in each row

edges_mean = np.mean(edges)

img_dims = gray.shape  # gray scale image dimensions
color_img_dims = (img_dims[0],img_dims[1],3)  # color image dimensions

clean_edges = np.zeros(color_img_dims,dtype=np.uint8)  # initialize array to hold edge values
clean_edges_gray = np.zeros(img_dims,dtype=np.uint8)  # initialize array to hold edge values

for i,val in enumerate(find_edges_0):  # iterate over columns
    # if the average column value is greater than the edges average + 1 standard deviation then consider this an edge

    if val > edges_mean + 1.25 * np.std(find_edges_0):
        clean_edges[:,i] = np.array([np.array([255,0,0]) for j in np.arange(edges.shape[0])],dtype=np.uint8)
        clean_edges_gray[:,i] = np.array([255 for j in np.arange(edges.shape[0])],dtype=np.uint8)
        img2_arr[:,i] = np.array([np.array([255,0,0]) for j in np.arange(edges.shape[0])],dtype=np.uint8)  # imposes edges on original image
    else:
        continue
    
for i,val in enumerate(find_edges_1):

    if val > edges_mean + 1.25 * np.std(find_edges_1):
        clean_edges[i,:] = np.array([np.array([255,0,0]) for j in np.arange(edges.shape[1])],dtype=np.uint8)
        clean_edges_gray[i,:] = np.array([255 for j in np.arange(edges.shape[1])],dtype=np.uint8)
        img2_arr[i,:] = np.array([np.array([255,0,0]) for j in np.arange(edges.shape[1])],dtype=np.uint8)
    else:
        continue

corners = cv.cornerHarris(clean_edges_gray,2,3,0.04)

#result is dilated for marking the corners, not important
corners_dialate = cv.dilate(corners,None)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]
img4[corners_dialate>0.01*corners_dialate.max()]=[0,0,255]
#clean_edges_gray[corners_dialate>0.01*corner_dialate.max()]=[0,0,255]
#img3 = cv.add(img2_arr,clean_edges)

##------------------------------------------------------------------------------------
#debug shit

#print(dst,dst.shape)
print(edges,edges.shape)

print(gray.shape)

##------------------------------------------------------------------------------------
#Show images

fig = plt.figure()

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img3)
ax2.imshow(clean_edges,cmap='gray')
ax3.imshow(img2_arr)

ax1.set_title('Original')
ax2.set_title('Edge Locations, Extended')
ax3.set_title('Extended Edges on Image')

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])

#plt.show()
##------------------------------------------------------------------------------------
fig = plt.figure()

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img3)
#ax2.imshow(dst)
ax2.imshow(edges,cmap='gray')
ax3.imshow(img)

ax1.set_title('Original')
ax2.set_title('Canny Edge Locations')
ax3.set_title('Canny Edges on Image')

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])

#plt.show()

##------------------------------------------------------------------------------------
fig = plt.figure()

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img3)
#ax2.imshow(dst)
ax2.imshow(corners_dialate,cmap='gray')
ax3.imshow(img4)

ax1.set_title('Original')
ax2.set_title('Corner Locations')
ax3.set_title('Corner Locations on Image')

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])

##------------------------------------------------------------------------------------
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(np.arange(find_edges_0.shape[0]),find_edges_0, label = 'Row Values')
ax2.scatter(np.arange(find_edges_1.shape[0]),find_edges_1, label = 'Column Values')

ax1.plot(np.arange(find_edges_0.shape[0]),[np.median(find_edges_0) for i in find_edges_0],color='r', label = 'Mean Row Values')
ax2.plot(np.arange(find_edges_1.shape[0]),[np.median(find_edges_1) for i in find_edges_1],color='r', label = 'Mean Column Values')


ax1.set_xlabel('Pixel')
ax2.set_xlabel('Pixel')

ax1.set_ylabel('Edge Value')
ax2.set_ylabel('Edge Value')

ax1.legend()
ax2.legend()

plt.show()
