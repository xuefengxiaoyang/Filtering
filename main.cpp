#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"

#include <iostream>
#include<vector>
#include<math.h>

using namespace std;
using namespace cv;


void PerformConvolution(Mat srcImg,Mat outImg,double **core,int size);
void ThreshodChoose(Mat filtered,int threshod);
void GetGaussianKernel(double **gaus, const int size,const double sigma);
void MergeImages(Mat xy_Img,Mat x_Img,Mat y_Img);
//访问图像像素点，实现阈值处理
int main()
{
    Mat srcImg1,srcImg2;
    srcImg1 = imread("G:/QTworkspace/experiment/balloonGrayNoisy.jpg",0);
    srcImg2 = imread("G:/QTworkspace/experiment/buildingGray.jpg",0);


    double **gaus=new double *[10];
    for(int i=0;i<10;i++)
    {
        gaus[i]=new double[10];  //动态生成矩阵
    }
/*#########################choose选择不同的参数，代表不同的操作，如下：#################################
                            0:sigma=1时，核的大小为3*sigma+1=4
                            1:sigma=2时，核的大小为3*sigma+1=7
                            2:sigma=3时，核的大小为3*sigma+1=10
                            3:卷积核大小为3的均值滤波
                            4:卷积核大小为5的均值滤波
                            5:sobel沿x,y轴方向和幅值的梯度边缘检测
                            6:prewitt沿x,y轴方向和幅值的梯度边缘检测

###################################################################################################*/
    int choose=6;
    switch (choose)
    {
        case 0:
        {
            Mat dstImg;
            dstImg = srcImg1.clone();//复制
            Mat outImg = Mat::zeros(dstImg.rows,dstImg.cols,dstImg.type());

            //高斯滤波，sigma等于1的时候。
            int sigma1=1;
            int size1=3*sigma1+1;
            //生成sigma=1时的高斯核
            GetGaussianKernel(gaus,size1,sigma1);
            PerformConvolution(dstImg,outImg,gaus,size1);
            cout<<"AAAAAAAAAAAAAAAAAAAA"<<endl;
            imshow("src", srcImg1);
            imshow("filtered", outImg);
            waitKey(0);
            break;
        }
        case 1:
        {
            Mat dstImg;
            dstImg = srcImg1.clone();//复制
            Mat outImg = Mat::zeros(dstImg.rows,dstImg.cols,dstImg.type());

            //高斯滤波，sigma等于2的时候。
            int sigma2=2;
            int size2=3*sigma2+1;
            //生成sigma=1时的高斯核
            GetGaussianKernel(gaus,size2,sigma2);
            PerformConvolution(dstImg,outImg,gaus,size2);
            imshow("src", srcImg1);
            imshow("filtered", outImg);
            waitKey(0);
            break;
        }
        case 2:
        {
            Mat dstImg;
            dstImg = srcImg1.clone();//复制
            Mat outImg = Mat::zeros(dstImg.rows,dstImg.cols,dstImg.type());

            //高斯滤波，sigma等于2的时候。
            int sigma3=3;
            int size3=3*sigma3+1;
            //生成sigma=1时的高斯核
            GetGaussianKernel(gaus,size3,sigma3);
            PerformConvolution(dstImg,outImg,gaus,size3);
            imshow("src", srcImg1);
            imshow("filtered", outImg);
            waitKey(0);
            break;
        }
        case 3:
        {
            Mat dstImg;
            dstImg = srcImg1.clone();//复制
            Mat outImg = Mat::zeros(dstImg.rows,dstImg.cols,dstImg.type());

            //均值滤波，size为3和5的时候，代表核的大小分别是3和5
            int size=3;
            double **core=new double*[size];
            for(int i=0;i<size;i++)
            {
                core[i] = new double[size];
            }
            for(int i=0;i<size;i++)
            {
                for(int j=0;j<size;j++)
                {
                    core[i][j]=1.0/(size*size);
                }
            }
            //滤波函数
            PerformConvolution(dstImg,outImg,core,size);
            imshow("src", srcImg1);
            imshow("filtered", outImg);
            waitKey(0);
            break;
        }
        case 4:
        {
            Mat dstImg;
            dstImg = srcImg1.clone();//复制
            Mat outImg = Mat::zeros(dstImg.rows,dstImg.cols,dstImg.type());

            //均值滤波，size为3和5的时候，代表核的大小分别是3和5
            int size=5;
            double **core=new double*[size];
            for(int i=0;i<size;i++)
            {
                core[i] = new double[size];
            }
            for(int i=0;i<size;i++)
            {
                for(int j=0;j<size;j++)
                {
                    core[i][j]=1.0/(size*size);
                }
            }
            PerformConvolution(dstImg,outImg,core,size);
            imshow("src", srcImg1);
            imshow("filtered", outImg);
            waitKey(0);
            break;
        }
        case 5:
        {
            Mat dstImgx,dstImgy,dstImgxy;
            dstImgx = srcImg2.clone();//复制
            dstImgy = srcImg2.clone();//复制
            dstImgxy = srcImg2.clone();//复制
            Mat outImgx = Mat::zeros(dstImgx.rows,dstImgx.cols,dstImgx.type());
            Mat outImgy = Mat::zeros(dstImgy.rows,dstImgy.cols,dstImgy.type());
            Mat outImgxy = Mat::zeros(dstImgxy.rows,dstImgxy.cols,dstImgxy.type());
            dstImgx = srcImg2.clone();//复制
            dstImgy = srcImg2.clone();//复制
            dstImgxy = srcImg2.clone();//复制
            int threshod =100;
            int sobel_y_size=3;
            double **corey=new double*[sobel_y_size];
            for(int i=0;i<sobel_y_size;i++)
            {
                corey[i] = new double[sobel_y_size];
            }

            int sobel_x_size=3;
            double **corex=new double*[sobel_x_size];
            for(int i=0;i<sobel_x_size;i++)
            {
                corex[i] = new double[sobel_x_size];
            }
            corey[0][0]=-1;
            corey[0][1]=-2;
            corey[0][2]=-1;
            corey[1][0]= 0;
            corey[1][1]= 0;
            corey[1][2]= 0;
            corey[2][0]= 1;
            corey[2][1]= 2;
            corey[2][2]= 1;

            corex[0][0]=-1;
            corex[0][1]=0;
            corex[0][2]=1;
            corex[1][0]=-2;
            corex[1][1]=0;
            corex[1][2]=2;
            corex[2][0]=-1;
            corex[2][1]=0;
            corex[2][2]=1;
            PerformConvolution(dstImgy,outImgy,corey,sobel_y_size);
            PerformConvolution(dstImgx,outImgx,corex,sobel_x_size);
            ThreshodChoose(outImgx,threshod);
            ThreshodChoose(outImgy,threshod);
            MergeImages(outImgxy,outImgxy,outImgx);
            imshow("src", srcImg2);
            imshow("x-filtered", outImgx);
            imshow("y-filtered", outImgy);
            imshow("xy-filtered", outImgy);
            waitKey(0);
            break;
        }

        case 6:
        {

            Mat dstImgx,dstImgy,dstImgxy;
            dstImgx = srcImg2.clone();//复制
            dstImgy = srcImg2.clone();//复制
            dstImgxy = srcImg2.clone();//复制
            Mat outImgx = Mat::zeros(dstImgx.rows,dstImgx.cols,dstImgx.type());
            Mat outImgy = Mat::zeros(dstImgy.rows,dstImgy.cols,dstImgy.type());
            Mat outImgxy = Mat::zeros(dstImgxy.rows,dstImgxy.cols,dstImgxy.type());

            int threshod =100;
            int prewitt_x_size=3;
            double **corex=new double*[prewitt_x_size];
            for(int i=0;i<prewitt_x_size;i++)
            {
                corex[i] = new double[prewitt_x_size];
            }

            int prewitt_y_size=3;
            double **corey=new double*[prewitt_y_size];
            for(int i=0;i<prewitt_y_size;i++)
            {
                corey[i] = new double[prewitt_y_size];
            }
            corex[0][0]=-1;
            corex[0][1]=0;
            corex[0][2]=1;
            corex[1][0]=-1;
            corex[1][1]=0;
            corex[1][2]=1;
            corex[2][0]=-1;
            corex[2][1]=0;
            corex[2][2]=1;

            corey[0][0]=1;
            corey[0][1]=1;
            corey[0][2]=1;
            corey[1][0]=0;
            corey[1][1]=0;
            corey[1][2]=0;
            corey[2][0]=-1;
            corey[2][1]=-1;
            corey[2][2]=-1;
            PerformConvolution(dstImgx,outImgx,corex,prewitt_x_size);
            PerformConvolution(dstImgy,outImgy,corey,prewitt_y_size);
            ThreshodChoose(outImgx,threshod);
            ThreshodChoose(outImgy,threshod);
            MergeImages(outImgxy,outImgx,outImgy);
            imshow("src", srcImg2);
            imshow("x-filtered", outImgx);
            imshow("y-filtered", outImgy);
            imshow("xy-filtered", outImgxy);
            waitKey(0);
            break;
            break;
        }
    }

    return 0;
}
//************************执行卷积的函数****************************
void PerformConvolution(Mat srcImg,Mat outImg,double **core,int size)
{
//    Mat dstImg;
//    dstImg = srcImg.clone();//复制
    int rows = srcImg.rows;
    int cols = srcImg.cols * srcImg.channels();

    for(int i=(size-1)/2;i<rows-(size-1)/2;i++)
    {
        for(int j=(size-1)/2;j<cols-(size-1)/2;j++)
        {
            float temp=0.0;
            for(int s=0;s<size;s++)
            {
                for(int v=0;v<size;v++)
                {
                    temp=temp+core[s][v]*srcImg.at<uchar>(i+s-(size-1)/2,j+v-(size-1)/2);
                }
            }
            outImg.at<uchar>(i,j)=(int)abs(temp);
        }

    }
}


//******************高斯卷积核生成函数*************************
void GetGaussianKernel(double **gaus, const int size,const double sigma)
{
    const double PI=4.0*atan(1.0); //圆周率π赋值
    int center=size/2;
    double sum=0;
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            gaus[i][j]=(1/(2*PI*sigma*sigma))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma));
            sum+=gaus[i][j];
        }
    }
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            gaus[i][j]/=sum;
        }
    }
}

//********************阈值选取函数********************************
void ThreshodChoose(Mat filtered,int threshod)
{
    int rows = filtered.rows;
    int cols = filtered.cols * filtered.channels();
    for(int j = 0; j < rows; j++)
    {
        unsigned char* data = filtered.ptr(j);//获取每行首地址

        for(int i = 0; i < cols; i++)
        {
            if(data[i]<threshod)
            {
                data[i]=0;
            }
            else
            {
                data[i]=255;
            }
        }
    }
}

//********************合并图像函数********************************
void MergeImages(Mat outImg,Mat x_Img,Mat y_Img)
{
    int rows = x_Img.rows;
    int cols = x_Img.cols * x_Img.channels();
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            outImg.at<uchar>(i,j)=x_Img.at<uchar>(i,j)+y_Img.at<uchar>(i,j);
        }
    }
}
