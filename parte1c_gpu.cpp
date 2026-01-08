#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // 1. CARGA DE IMAGEN (Usa una de tus imágenes de la práctica)
    Mat img = imread("datasets/person_data/images/ds2_pexels-photo-551659.png");
    if (img.empty()) { 
        cout << "Error al cargar imagen." << endl; 
        return -1; 
    }

    Mat gray, gaussian, morphology, edges, hist_equalized;

    // --- OPERACIÓN 1: Suavizado (Filtro Gaussiano) ---
    GaussianBlur(img, gaussian, Size(15, 15), 0);

    // --- OPERACIÓN 2: Detección de bordes (Canny) ---
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 100, 200);

    // --- OPERACIÓN 3: Opacidades Morfológicas (Dilatación) ---
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(edges, morphology, kernel);

    // --- OPERACIÓN 4: Ecualización del Histograma ---
    equalizeHist(gray, hist_equalized);

    // MOSTRAR RESULTADOS (Toma captura de esto para la Parte 1C)
    imshow("1. Original", img);
    imshow("2. Suavizado (Gauss)", gaussian);
    imshow("3. Bordes (Canny)", edges);
    imshow("4. Morfologia (Dilatacion)", morphology);
    imshow("5. Histograma Ecualizado", hist_equalized);

    cout << "--- Analisis de Rendimiento ---" << endl;
    cout << "Pipeline actual: CPU-based." << endl;
    cout << "Pipeline propuesto en guia: GPU-only (CUDA)." << endl;

    waitKey(0);
    return 0;
}