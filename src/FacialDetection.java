import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FacialDetection {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//		System.out.println("cchec");
		String imgFile = "images/test.jpg";
		Mat src = Imgcodecs.imread(imgFile);
		
		String xmlFile = "xml/lbpcascade_frontalface.xml";
		CascadeClassifier cc = new CascadeClassifier(xmlFile);
		
		//We do this by first applying one cascade classifier scan over the the entire image 
		//to give us an MatOfRect object containing our large object
		MatOfRect faceDetection = new MatOfRect(); 
		
		//using classifier to read src file and MatOfRect Object.
		cc.detectMultiScale(src, faceDetection);
		System.out.println(String.format("Face Found: %d", faceDetection.toArray().length)); //.toArray detects all the faces
		/*
		 * We then iterate over the Rect[] array given by the toArray() function from the MatOfRect object. 
		 * This Rect object is used in creating a temporary Mat object that is "cropped" to the Rect object's properties 
		 * (x, y, width, height) from the original image, where we can then perform detections on the temporary Mat object. 
		 */
		for(Rect rect: faceDetection.toArray()) {
			Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0,0,255),3);
		}
	
		Imgcodecs.imwrite("images/test-out.jpg", src);
		System.out.println("Image Detected Successfully!");
	}
}
