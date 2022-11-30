package frm;

import org.encog.Encog;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.MLResettable;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.temporal.TemporalDataDescription;
import org.encog.ml.data.temporal.TemporalMLDataSet;
import org.encog.ml.data.temporal.TemporalPoint;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.factory.MLTrainFactory;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.RequiredImprovementStrategy;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

import java.io.File;

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
    //public static final int INPUT_WINDOW_SIZE = 1;

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

    /**
     * Create a temporal data set to train the network with
     * @param rawFile the raw input file object
     * @return temporal data set
     */
    public static TemporalMLDataSet createTraining(File rawFile) {
        TemporalMLDataSet trainingData = initDataSet();
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

    /**
     * Create and train a model using Encog factory codes to specify the model type.
     * @param trainingData The temporal training data set
     * @param methodName
     * @param methodArchitecture
     * @param trainerName
     * @param trainerArgs
     * @return
     */
    public static MLRegression trainModel(
            MLDataSet trainingData,
            String methodName,
            String methodArchitecture,
            String trainerName,
            String trainerArgs) {

        // first, create the machine learning method (the model)
        MLMethodFactory methodFactory = new MLMethodFactory();
        MLMethod method = methodFactory.create(
                methodName,
                methodArchitecture,
                trainingData.getInputSize(),
                trainingData.getIdealSize()
        );

        // second, create the trainer
        MLTrainFactory trainFactory = new MLTrainFactory();
        MLTrain train = trainFactory.create(method, trainingData, trainerName, trainerArgs);

        // reset if improve is less than 1% over 5 cycles
        if( method instanceof MLResettable && !(train instanceof ManhattanPropagation) ) {
            train.addStrategy(new RequiredImprovementStrategy(500));
        }

        // third train the model
        EncogUtility.trainToError(train, 0.002);

        return (MLRegression)train.getMethod();
    }

    public static TemporalMLDataSet predict(File rawFile, MLRegression model) {
        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ',');
        while (csv.next()) {
            // Date format is yyyy-mm-dd, we'll pare it into yyyymmdd to be used as a sequence integer
            String dateString = csv.get(0);
            String modifiedDate = dateString.replace("-", "");
            int sequenceNumber = Integer.parseInt(modifiedDate);

            double interestRate = csv.getDouble(1);

            // do we have enough data for a prediction yet?
            if (trainingData.getPoints().size() >= trainingData.getInputWindowSize()) {
                // Make sure to use index 1, because the temporal data set is always one ahead
                // of the time slice its encoding.  So for RAW data we are really encoding 0.
                MLData modelInput = trainingData.generateInputNeuralData(1);
                MLData modelOutput = model.compute(modelInput);
                double predictedRate = normRate.deNormalize(modelOutput.getData(0));
                System.out.println("Date: " + sequenceNumber + " Predicted= " + predictedRate + " Actual= " + interestRate);

                // Remove the earliest training element.  Unlike when we produced training data,
                // we do not want to build up a large data set.  We just add enough data points to produce
                // input to the model.
                trainingData.getPoints().remove(0);
            }

            // Add the next point to the temporal data set
            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(sequenceNumber);
            point.setData(0, normRate.normalize(interestRate));
            trainingData.getPoints().add(point);
        }
        csv.close();

        // generate the time-boxed data
        trainingData.generate();
        return trainingData;
    }

    /**
     * The main method.
     * @param args The arguments.
     */
    public static void main(String[] args) {
        // Get the raw mortgage rate training data
        File rawFile = new File(MYDIR, "MORTGAGE30US.csv");

        // Build the training data set
        TemporalMLDataSet trainingData = createTraining(rawFile);

        // Build the model and train
        MLRegression model = trainModel(
                trainingData,
                MLMethodFactory.TYPE_FEEDFORWARD,
                "?:B->SIGMOID->25:B->SIGMOID->?",
                MLTrainFactory.TYPE_RPROP,
                "");

        predict(rawFile, model);

        Encog.getInstance().shutdown();

    }

}
