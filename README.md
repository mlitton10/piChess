# piChess
Capture real time images of chess games and convert them to into FEN strings for stockfish analysis

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


5. We can overlay this new grid ontop of the original image:
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/board_with_clean_edges.jpeg?raw=true)

6. Next, we can use this grid to find corners in the image using some algorithm. I tried out Harris Corner detection for now but there are a few options
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/corners_only_image.jpeg?raw=true)

7.Finally, we can overlay these corners onto the original image:
![alt text](https://github.com/mlitton10/piChess/blob/main/test_images/board_with_corners.jpeg?raw=true)
