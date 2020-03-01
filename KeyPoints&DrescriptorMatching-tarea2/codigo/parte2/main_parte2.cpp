#define _DEBUG
#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include "sift.hpp"

using namespace std;
using namespace cv;

void do_match(Mat queryDescriptors, Mat trainDescriptors, std::vector<DMatch>& matches)
{
	for (int i = 0; i < queryDescriptors.rows; i++)
	{
		DMatch match; 
		match.imgIdx = 0;
		match.queryIdx = i;
		float distancia_minima = float(1000000.0);//numero excesivamente grande 
		for (int j = 0; j < trainDescriptors.rows; j++)
		{
			float distancia_entre_descriptoresij = float(0.0);
			for (int k=0; k<128; k++) 
			{ 
				float d;
				d = queryDescriptors.at<float>(i, k) - trainDescriptors.at<float>(j, k); 
				distancia_entre_descriptoresij += d*d; 
			}
			distancia_entre_descriptoresij = sqrt(distancia_entre_descriptoresij/128);
			
			if(distancia_entre_descriptoresij < distancia_minima)
			{
				distancia_minima = distancia_entre_descriptoresij;
				match.trainIdx = j;
				match.distance = distancia_minima;
			}
		}
		matches.push_back(match);
	}
	cout << "queryDescriptors rows=" << queryDescriptors.rows << " cols=" << queryDescriptors.cols << endl;
}

int main(void)
{
	string train_img_names[] = {"dentifrice", "ice1" , "uch006a" , "uch084a","uch098a"}; 
	string query_img_names[] = {"dentifrice2", "ice2" , "uch006b" , "uch084b","uch098b"}; 
	for (int i=0; i<5 ;i++)
	{
		Mat input1, input2;
		input1 = imread("im/"+train_img_names[i]+".jpg"); // TRAIN image
		input2 = imread("im/"+query_img_names[i]+".jpg"); //QUERY image

		srand(time(NULL)); // Inicializar generador de numeros al azar

		if(input1.empty() || input2.empty()) // No encontro la imagen
		{
			cout<<"Imagen no encontrada"<<endl;
			return 1; // Sale del programa anormalmente
		}

		vector<KeyPoint> keypoints1;
		vector<KeyPoint> keypoints2;
		Mat descriptors1, descriptors2;

		Ptr<DescriptorExtractor> descriptorExtractor = xfeatures2d::SIFT::create(100);
		descriptorExtractor->detectAndCompute(input1, Mat(), keypoints1, descriptors1);
		descriptorExtractor->detectAndCompute(input2, Mat(), keypoints2, descriptors2);

		cout << "Keypoints im 1 " << keypoints1.size() << endl;
		cout << "Keypoints im 2 " << keypoints2.size() << endl;

		vector<DMatch> matches;
		do_match(descriptors2, descriptors1, matches);

		// Dibujar resultados
		Mat output1, output2;
		drawKeypoints(input1, keypoints1, output1);
		imwrite(train_img_names[i]+"_keypoints"+".jpg", output1);
		drawKeypoints(input2, keypoints2, output2);
		imwrite(query_img_names[i]+"_keypoints"+".jpg", output2);

		Mat img_matches;
		drawMatches(input2, keypoints2, input1, keypoints1, matches, img_matches);
		imwrite(train_img_names[i]+"_matches"+".jpg", img_matches);
	}
	cout<<"*************************************************************************************"<<endl;
	cout<<"   El codigo funciono y guardo todas las imagenes resultantes en la carpeta bluid"<<endl;	
	cout<<"*************************************************************************************"<<endl;
	waitKey(0);
	return 0; // Sale del programa normalmente
}
