import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

##------------------------------------------------------------------------------------
# Read the image files in

img_dir = 'test_images/'
img_file = 'blank_board_2.jpeg'
#img_file = 'skewed_board.jpeg'

img_loc = ''.join([img_dir,img_file])
img = cv.imread(img_loc)
img2 = cv.imread(img_loc)
img3 = cv.imread(img_loc)
img4 = cv.imread(img_loc)
hough_line = cv.imread(img_loc)
##------------------------------------------------------------------------------------

# Image processing

# Create gray scale image and vectorize it
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.uint8(gray)

img2_arr = np.uint8(img2) # turn image into unsigned 8 bit integer array
#print(img2_arr.shape)

# find edges with the Canny edge detector algo
edges = cv.Canny(gray,100,200)

img_dims = gray.shape  # gray scale image dimensions
color_img_dims = (img_dims[0],img_dims[1],3)  # color image dimensions

clean_edges = np.zeros(color_img_dims,dtype=np.uint8)  # initialize array to hold edge values
clean_edges_gray = np.zeros(img_dims,dtype=np.uint8)  # initialize array to hold edge values

edge_rows = []
edge_columns = []

def get_edges_from_canny_1(image):
    
    def find_edges(mean_val,axis_mean):
        axis_edge = []
        for i,val in enumerate(axis_mean):  # iterate over columns
            # if the average column value is greater than the edges average + 1 standard deviation then consider this an edge
            
            if val > mean_val + 1.25 * np.std(axis_mean):
                axis_edge.append(i)
            else:
                continue
        return axis_edge

        
    find_edges_0 = np.mean(image,axis=0)  # find the average value of pixel in each column
    find_edges_1 = np.mean(image,axis=1)  # find avearge value of pixel in each row
    edges_mean = np.mean(image)

    edge_columns = find_edges(edges_mean, find_edges_0)
    edge_rows = find_edges(edges_mean, find_edges_1)

    print(edge_columns,edge_rows)
    def clean_edges(edge_axis):
        avg_column = []
        max_ind = -1
        for val in edge_axis:
            if val<=max_ind:
                continue
            else:
                sub_arr = []
                while val in edge_axis:
                    sub_arr.append(val)
                    val+=1
                avg_val = int(np.mean(sub_arr))
                avg_column.append(avg_val)
                max_ind = val+1
        return avg_column

    clean_columns = clean_edges(edge_columns)
    clean_rows = clean_edges(edge_rows)

    return clean_columns, clean_rows

## find exact edges, columns:

edge_columns,edge_rows = get_edges_from_canny_1(edges)
print(edge_columns)
print(edge_rows)


def build_edge_image(row_locs,col_locs,image_container):
    img_dims = image_container.shape  # gray scale image dimensions
    color_img_dims = (img_dims[0],img_dims[1],3)  # color image dimensions

#    print(img_dims)
    clean_edges = np.zeros(color_img_dims,dtype=np.uint8)  # initialize array to hold edge values
    clean_edges_gray = np.zeros((img_dims[0],img_dims[1]),dtype=np.uint8)  # initialize array to hold edge values

 #   print(len(col_locs),len(clean_edges[:,0]))
    for i in col_locs:  # iterate over columns
        clean_edges[:,i] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[0])],dtype=np.uint8)
        clean_edges_gray[:,i] = np.array([255 for j in np.arange(img_dims[0])],dtype=np.uint8)
        image_container[:,i] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[0])],dtype=np.uint8)  # imposes edges on original image
        try:
            image_container[:,i+1] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[0])],dtype=np.uint8)
        except Exception as e:
            print(e)
        try:
            image_container[:,i-1] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[0])],dtype=np.uint8)
        except Exception as e:
            print(e)
  #  print(len(row_locs))
    for i in row_locs:
        clean_edges[i,:] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[1])],dtype=np.uint8)
        clean_edges_gray[i,:] = np.array([255 for j in np.arange(img_dims[1])],dtype=np.uint8)
        image_container[i,:] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[1])],dtype=np.uint8)
        try:
            image_container[i+1,:] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[1])],dtype=np.uint8)
        except Exception as e:
            print(e)
        try:
            image_container[i-1,:] = np.array([np.array([255,0,0]) for j in np.arange(img_dims[1])],dtype=np.uint8)
        except Exception as e:
            print(e)

    return clean_edges, clean_edges_gray, image_container


clean_edges, clean_edges_gray, img2_arr = build_edge_image(edge_rows, edge_columns,img2_arr)

hough_lines = cv.HoughLines(edges,1,1/180,70,20)
print(len(hough_lines))
hough_lines_only = np.zeros(color_img_dims,dtype=np.uint8)  # initialize array to hold edge values

for line in hough_lines:
    x1, y1, x2, y2 = line[0]
    cv.line(hough_line, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv.line(hough_lines_only, (x1, y1), (x2, y2), (255, 0, 0), 3)

corners = cv.cornerHarris(clean_edges_gray,2,3,0.04)



#result is dilated for marking the corners, not important
corners_dialate = cv.dilate(corners,None)

corners_dialate_color = np.zeros(color_img_dims,np.uint8)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]
img4[corners_dialate>0.01*corners_dialate.max()]=[0,0,255]
corners_dialate_color[corners_dialate>0.01*corners_dialate.max()]=[255,0,0]
#clean_edges_gray[corners_dialate>0.01*corner_dialate.max()]=[0,0,255]
#img3 = cv.add(img2_arr,clean_edges)

##------------------------------------------------------------------------------------
#debug shit

#print(dst,dst.shape)
#print(edges,edges.shape)

#print(gray.shape)

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

cv.imwrite(''.join([img_dir,'board_with_clean_edges.jpeg']), img2_arr)
cv.imwrite(''.join([img_dir,'clean_edges_only.jpeg']), clean_edges)

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])


fig = plt.figure()

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img3)
ax2.imshow(hough_lines_only,cmap='gray')
ax3.imshow(hough_line)

ax1.set_title('Original')
ax2.set_title('Edge Locations, Extended')
ax3.set_title('Extended Edges on Image')

#cv.imwrite(''.join([img_dir,'board_with_clean_edges.jpeg']), img2_arr)
#
cv.imwrite(''.join([img_dir,'board_hough_lines.jpeg']), hough_lines)
cv.imwrite(''.join([img_dir,'hough_lines_only.jpeg']), hough_lines)

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

cv.imwrite(''.join([img_dir,'board_canny_edges.jpeg']), img)
cv.imwrite(''.join([img_dir,'canny_edges_only.jpeg']), edges)

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

cv.imwrite(''.join([img_dir,'board_with_corners.jpeg']), img4)
cv.imwrite(''.join([img_dir,'corners_only_image.jpeg']), corners_dialate_color)

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])

##------------------------------------------------------------------------------------
#fig = plt.figure()
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)

#ax1.scatter(np.arange(clean_rows.shape[0]),find_edges_0, label = 'Row Values')
#ax2.scatter(np.arange(clean_edges.shape[0]),find_edges_1, label = 'Column Values')

#ax1.plot(np.arange(clean_rows.shape[0]),[np.median(find_edges_0) for i in find_edges_0],color='r', label = 'Mean Row Values')
#ax2.plot(np.arange(clean_edges.shape[0]),[np.median(find_edges_1) for i in find_edges_1],color='r', label = 'Mean Column Values')


#ax1.set_xlabel('Pixel')
#ax2.set_xlabel('Pixel')

#ax1.set_ylabel('Edge Value')
#ax2.set_ylabel('Edge Value')

#ax1.legend()
#ax2.legend()

plt.show()
