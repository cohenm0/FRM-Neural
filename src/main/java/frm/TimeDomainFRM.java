package frm;

import org.encog.bot.BotUtil;
import org.encog.ml.data.temporal.TemporalDataDescription;
import org.encog.ml.data.temporal.TemporalMLDataSet;
import org.encog.ml.data.temporal.TemporalPoint;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.csv.ReadCSV;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Date;

public class TimeDomainFRM {
    /**
     * Set this to whatever you want to use as your home directory.
     * The example is set to use the current directory.
     */
    public static final File MYDIR = new File("data/");

    /**
     * This is the amount of data to use to guide the prediction.
     */
    public static final int INPUT_WINDOW_SIZE = 12;

    /**
     * This is the amount of data to actually predict.
     */
    public static final int PREDICT_WINDOW_SIZE = 1;

    /**
     * Normalize the mortgage rate values to 0-1.
     */
    public static NormalizedField normRate = new NormalizedField(
            NormalizationAction.Normalize, "rate", 18.63, 2.65, 1, 0);

    public static TemporalMLDataSet initDataSet() {
        // create a temporal data set
        TemporalMLDataSet dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);

        // We set both input and predict flags to true since we are using the same dataset for both
        TemporalDataDescription mortgageRates = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true);

        dataSet.addDescription(mortgageRates);
        return dataSet;
    }

    //public static MLRegression trainModel(

    public static TemporalMLDataSet createTraining(File rawFile) {
        TemporalMLDataSet trainingData = initDataSet();
        String s = rawFile.toString();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ',');
        while (csv.next()) {
            // Date format is yyyy-mm-dd, we'll pare it into yyyymmdd to be used as a sequence integer
            String dateString = csv.get(0);
            String modifiedDate = dateString.replace("-", "");
            int sequenceNumber = Integer.parseInt(modifiedDate);

            double interestRate = csv.getDouble(1);

            TemporalPoint point = new TemporalPoint(trainingData
                    .getDescriptions().size());
            point.setSequence(sequenceNumber);
            point.setData(0, normRate.normalize(interestRate));
            trainingData.getPoints().add(point);
        }
        csv.close();

        // generate the time-boxed data
        trainingData.generate();
        return trainingData;
    }

    //public static TemporalMLDataSet predict(File rawFile, MLRegression model)

    /**
     * The main method.
     * @param args The arguments.
     */
    public static void main(String[] args) {
        File rawFile = new File(MYDIR, "data/MORTGAGE30US.csv");


    }

}
