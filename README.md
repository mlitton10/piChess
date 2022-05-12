# piChess
Capture real time images of chess games and convert them to into FEN strings for stockfish analysis

A good paper:
https://mdpi-res.com/d_attachment/jimaging/jimaging-07-00094/article_deploy/jimaging-07-00094-v2.pdf?version=1622796868

There is a major problem evident in this paper:
  See page 14 figure 12
  Mean inference time on CPU: 2.11 +- 0.64 s
  Mean inference time on GPU: 0.35 +- 0.06 s
The CPU inference time is slow, and we can assume that the rpi4 will be even slower. The CPU used was a 3.20 GHz Intel Core i5-6500 CPU. Rpi is not this good. Most of this time is taken up by the piece classification CNN.

Current State:

1. Start with image of chess board:
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/blank_board_2.jpeg?raw=true)

2. Use Canny edge detection to find true edges. This finds the edges in the image but they are 'wobbly' and also don't extend all the way to the end of the image which could be useful for cropping. 
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/canny_edges_only.jpeg?raw=true)

3. To improve this I used statistical methods to find the pixel locations where the edges occured and created an image of these edges extended through the whole image. Again, this could be useful for cropping and defining board coordinate systems. It also may be helpful for removing the extraneous stuff like the row/column letter number labels. This is NOT a Hough transform but achieves a very similiar end result.
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/clean_edges_only.jpeg?raw=true)

Methodology:
First we find the average gray scale pixel value in each row and column of the image:
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\left<&space;Pixel_{row,i}\right>&space;=&space;\frac{1}{N_{col}}\sum_{j=1}^{N_{col}}Pixel(i,j)&space;)
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\left<&space;Pixel_{column,j}\right>&space;=&space;\frac{1}{N_{row}}\sum_{i=1}^{N_{row}}Pixel(i,j)&space;)

The overall mean pixel value of the image:
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\left<&space;Pixel\right>&space;=&space;\frac{1}{N_{row}N_{col}}\sum_{i=1}^{N_{row}}\sum_{j=1}^{N_{col}}Pixel(i,j)&space;)

And the standard deviations of rows and columns:
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\sigma_{row,i}&space;=&space;\sqrt{\frac{1}{N_{col}}\sum_{j=1}^{N_{col}}Pixel(i,j)-\left<Pixel_{row,i}\right>}&space;)
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\sigma_{column,j}&space;=&space;\sqrt{\frac{1}{N_{row}}\sum_{i=1}^{N_{row}}Pixel(i,j)-\left<Pixel_{column,j}\right>}&space;)

We consider a row/column to be an edge if the following condition is true:
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\left<&space;Pixel_{row,i}\right>&space;>&space;\left<&space;Pixel&space;\right>&space;&plus;&space;1.25&space;\sigma_{row,i})
![equation](https://latex.codecogs.com/png.image?\dpi{120}\bg{white}\left<&space;Pixel_{column,j}\right>&space;>&space;\left<&space;Pixel&space;\right>&space;&plus;&space;1.25&space;\sigma_{column,j})


4. We can overlay this new grid ontop of the original image:
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/board_with_clean_edges.jpeg?raw=true)

5. Next, we can use this grid to find corners in the image using some algorithm. I tried out Harris Corner detection for now but there are a few options
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/corners_only_image.jpeg?raw=true)


6.Finally, we can overlay these corners onto the original image:
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/board_with_corners.jpeg?raw=true)


The above methodology will fail for skewed images or images where the board is rotated in frame. The failure occurs at step 3 since the statistical analysis assumes the camera is top down view. The error will scale with rotation angle or skew. Example of this effect:


We can fix this problem by using a Hough transform instead.
