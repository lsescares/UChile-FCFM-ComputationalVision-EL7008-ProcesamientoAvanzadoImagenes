#define _DEBUG
#include <opencv2/opencv.hpp>
#include "sift.hpp"
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Mat do_downsample(Mat input)
{
	Mat output = input.clone();
	pyrDown(input, output);//por defecto escala los tamaños a la mitad
	return output;
}

Mat do_rotate(Mat input)
{
	//FUENTE:https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
	//Linea a linea se explica en el informe.
	Mat output = input.clone();
	double angle = 30;
    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((input.cols-1)/2.0, (input.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), input.size(), angle).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - input.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - input.rows/2.0;
    cv::warpAffine(input, output, rot, bbox.size());

	return output;
}


int main(void){
	string img_names[] = {"casa","dibujo","figurasllenas"}; 
	for (int s=0;s<3;s++){
		Mat input, scaled, rotated; 
		input = imread(img_names[s]+".png"); 
		if (input.empty()){
			cout<< "la imagen "+img_names[s]+" no fue encontrada" << endl;
			return 1;
		}

		scaled = do_downsample(input);
		imwrite(img_names[s]+"_scaled.jpg",scaled);
		rotated = do_rotate(input);
		imwrite(img_names[s]+"_rotated.jpg",rotated);
		
		if (scaled.empty()){
			cout << "Error en el down_sample" << endl;
			return 2; 
		}
		if (rotated.empty()) {
			cout << "Error en la rotacion" << endl;
			return 2; 
		}
		
		srand(time(NULL)); // Inicializar generador de numeros al azar

		vector<KeyPoint> keypoints1;
		vector<KeyPoint> keypoints2;
		vector<KeyPoint> keypoints3;
		//############################################################################################
		//--Comentar y descomentar estos bloqes para seleccionar el metodos identificados de keypoints a usar-----------
		//Ptr<FeatureDetector> featureDetector = ORB::create();
		//string name_keyPointsDetector = "ORB";  
	
		//Ptr<FeatureDetector> featureDetector = xfeatures2d::SIFT::create();
		//string name_keyPointsDetector = "SIFT";
			
		Ptr<FeatureDetector> featureDetector = GFTTDetector::create();  // Good Features to Track
		string name_keyPointsDetector = "GFTT";
		//###########################################################################################
		
		featureDetector->detect(input, keypoints1);
		featureDetector->detect(scaled, keypoints2);
		featureDetector->detect(rotated, keypoints3);
		// Las regiones obtenidas por ORB son muy grandes al visualizarlas, se deben reducir, ignorar esto
		
		//---------------------------------------------------------------------------------
		//Cuando se elije detector ORB, este bloque if no se ejecuta. Entonces, comentar la condicion, es decir,
		//dejar que siempre se ejecuten esos 3 for, para achicar el tamaño de los circulos de ORB.
		if (featureDetector->getDefaultName() == cv::String("Feature2D.ORB"))
		{
			for (int i = 0; i < keypoints1.size(); i++)
				keypoints1[i].size = keypoints1[i].size / 5;
			for (int i = 0; i < keypoints2.size(); i++)
				keypoints2[i].size = keypoints2[i].size / 5;
			for (int i = 0; i < keypoints3.size(); i++)
				keypoints3[i].size = keypoints3[i].size / 5;
		}
		//------------------------------------------------------------------------------------
		cout<<"IMAGEN= "<<img_names[s]<<", detector ptos interes= "<<name_keyPointsDetector<<endl;
		cout<<"Keypoints imagen original = "<<keypoints1.size()<<endl;
		cout<<"Keypoints imagen escalada  = "<<keypoints2.size()<<endl;
		cout<<"Keypoints imagen rotada = "<<keypoints3.size()<<endl;

		Mat output_original, output_scaled, output_rotated;
		drawKeypoints(input, keypoints1, output_original, 0,DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(scaled, keypoints2, output_scaled, 0, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(rotated, keypoints3, output_rotated, 0, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imwrite(img_names[s]+"_"+name_keyPointsDetector+"_keypoints1.jpg",output_original);
		imwrite(img_names[s]+"_"+name_keyPointsDetector+"_keypoints2.jpg",output_scaled);
		imwrite(img_names[s]+"_"+name_keyPointsDetector+"_keypoints3.jpg",output_rotated);	
	}	
	cout<<"*************************************************************************************"<<endl;
	cout<<"   El codigo funciono y guardo todas las imagenes resultantes en la carpeta bluid"<<endl;	
	cout<<"*************************************************************************************"<<endl;
	waitKey(0);
	return 0; // Sale del programa normalmente
}
