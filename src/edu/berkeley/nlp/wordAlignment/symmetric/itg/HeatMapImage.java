package edu.berkeley.nlp.wordAlignment.symmetric.itg;

import edu.berkeley.nlp.math.DoubleArrays;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class HeatMapImage {
  private double[][] data;
  private int[][] dataColorIndices;

  private Color[] colors;
  public int pixelSize = 20;

  private BufferedImage bufferedImage;
  private Graphics2D bufferedGraphics;

  public HeatMapImage(boolean useGraphicsYAxis, Color[] colors)
  {
      updateGradient(colors);
  }
  
  public HeatMapImage()
  {
    this(false,Gradient.GRADIENT_BLACK_TO_WHITE);
  }

  public void writePNGImage(double[][] data, String outfile)
  {
    try {
      updateData(data,true);
      updateDataColors();
      drawData();
      ImageIO.write(bufferedImage,"png",new File(outfile));
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void updateGradient(Color[] colors)
  {
      this.colors = colors.clone();
   }

  private void updateDataColors()
  {
//    for (double[] row : data) {
//      for (int i = 0; i < row.length; i++) {
//        double v = row[i];
//        row[i] = Math.exp(v);
//      }
//    }

	for (double[] row: data) {
		for (int i = 0; i < row.length; ++i) {
			row[i] = Double.parseDouble(String.format("%1.4f ", row[i]));
		}
	}
	double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;
    for (double[] row : data) {
      max = Math.max(max,DoubleArrays.max(row));
      min = Math.min(min, DoubleArrays.min(row));
    }
    // max = 1.0/data.length;
    // min = 0.0;
    double range = max - min;
    dataColorIndices = new int[data.length][data[0].length];
      //assign a Color to each data point
      for (int x = 0; x < data.length; x++)
      {
          for (int y = 0; y < data[0].length; y++)
          {
            double p = (data[x][y]-min) / range;
            int colorIndex = (int) Math.floor(p * (colors.length - 1));
            if (colorIndex >= colors.length) colorIndex = colors.length-1;
            dataColorIndices[x][y] = colorIndex;
          }
      }
  }

  public void updateData(double[][] data, boolean useGraphicsYAxis)
  {
      this.data = new double[data.length][data[0].length];
      for (int ix = 0; ix < data.length; ix++)
      {
          for (int iy = 0; iy < data[0].length; iy++)
          {
              if (useGraphicsYAxis)
              {
                  this.data[ix][iy] = data[ix][iy];
              }
              else
              {
                  this.data[ix][iy] = data[ix][data[0].length - iy - 1];
              }
          }
      }

//      System.out.println();
//      System.out.println("HeatMapImage data size: "+data.length+"-"+data[0].length);
//      for (int ix = 0; ix < data.length; ++ix) {
//          for (int iy = 0; iy < data[0].length; ++iy) {
//        	  System.out.print(String.format("%1.4f ", data[ix][iy]));
//          }
//          System.out.println();
//      }
      
  }

  private void drawData()
  {
      bufferedImage = new BufferedImage(pixelSize *data.length, pixelSize *data[0].length, BufferedImage.TYPE_INT_ARGB);
      bufferedGraphics = bufferedImage.createGraphics();

      for (int x = 0; x < data.length; x++)
      {
          for (int y = 0; y < data[0].length; y++)
          {
              bufferedGraphics.setColor(colors[dataColorIndices[x][y]]);
              bufferedGraphics.fillRect(pixelSize * x, pixelSize * y, pixelSize, pixelSize);
          }
      }
  }

  public static void main(String[] args) {
    int n = 100;
    java.util.Random rand = new java.util.Random();
    double[][] m = new double[n][n];
    for (int i=0; i < n;  ++i)  {
      for (int j=0; j < n; ++j) {
        m[i][j] = rand.nextDouble();
      }
    }
    HeatMapImage hmi = new HeatMapImage();
    hmi.writePNGImage(m,"out.png");
  }

}