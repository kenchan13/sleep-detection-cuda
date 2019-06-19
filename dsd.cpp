#include <unistd.h>
#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>

using namespace std;
using namespace cv;
//using namespace cuda;

Ptr<cuda::CascadeClassifier> face_cascade_gpu;
Ptr<cuda::CascadeClassifier> eye_cascade_gpu;
VideoCapture capture;

int main(int argc, char **argv)
{
    int t = 0;
    
    cout << "cuda::getCudaEnabledDeviceCount(): " << cuda::getCudaEnabledDeviceCount() << std::endl;
    cuda::printCudaDeviceInfo(0);
    cuda::setDevice(0);

    face_cascade_gpu = cuda::CascadeClassifier::create("haarcascade_frontalface_default.xml");
    eye_cascade_gpu = cuda::CascadeClassifier::create("haarcascade_eye.xml");
    capture.open(0);
    
    Mat frame;
    while (capture.read(frame))
    {
        bool sleeping_detected = false;
        cuda::GpuMat frame_gpu(frame);
        cuda::GpuMat gray, objbuf;
        cuda::cvtColor(frame_gpu, gray, COLOR_BGR2GRAY, 0);
        
        face_cascade_gpu->setScaleFactor(1.3);
        face_cascade_gpu->setMinNeighbors(5);
        face_cascade_gpu->detectMultiScale(gray, objbuf);
        
        vector<Rect> faces;
        face_cascade_gpu->convert(objbuf, faces);
        
        int n = faces.size(), m;
        //vector<cuda::Stream> streams(n);
        vector<cuda::GpuMat> roi_gray(n);
        //vector<cuda::GpuMat> roi_color(n);
        vector<cuda::GpuMat> eyes_detected_objbuf(n);
        for(int i = 0; i < n; i++)
        {
            //streams[i] = cuda::Stream();
            roi_gray[i] = cuda::GpuMat(gray, faces[i]);
            //roi_color[i] = cuda::GpuMat(frame_gpu, faces[i]);
            
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);
            
            eye_cascade_gpu->setScaleFactor(1.1);
            eye_cascade_gpu->setMinNeighbors(3);
            //cuda::GpuMat eyebuf;
            //eye_cascade_gpu->detectMultiScale(roi_gray[i], eyes_detected_objbuf[i], streams[i]);
            eye_cascade_gpu->detectMultiScale(roi_gray[i], eyes_detected_objbuf[i]);

            vector<Rect> eyes;
            eye_cascade_gpu->convert(eyes_detected_objbuf[i], eyes);

            m = eyes.size();
            for (int j = 0; j < m; j++)
            {
                rectangle(frame, Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar(0, 255, 0), 2);
            }
            
            if (m == 0)
            {
                if ((++t) > 5)
                {
                    sleeping_detected = true;
                }
            }
        }
        
        if (sleeping_detected)
        {
            time_t ti = time(NULL);
            char tibuf[64];
            cout << strftime(tibuf, sizeof(tibuf), "[%Y-%m-%d %H:%M:%S] ", localtime(&ti)) << "driver is sleeping !\n";
        }
        else
        {
            t = 0;
        }
        
        imshow("frame", frame);
        waitKey(1);
    }
    
    return 0;
}
