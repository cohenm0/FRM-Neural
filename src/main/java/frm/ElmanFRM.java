package frm;

import org.encog.bot.browse.range.Input;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.csv.ReadCSV;
import java.io.File;

public class ElmanFRM {

    public static final File MYDIR = new File("data/");

    public static File rawFile = new File(MYDIR, "MORTGAGE30US.csv");

    public final static double[] RATES = getRates(rawFile);

    public final static int[] DATES = getDates(rawFile);

    public final static int TRAIN_START = 0;

    public final static int TRAIN_END = 2299;

    public final static int TESTING_START = 2300;

    public final static int EVALUATE_END = RATES.length-1;

    public final static double MAX_ERROR = 0.001;

    public final static double LOW = 0.1;

    public final static double HIGH = 0.9;

    // This is the amount of data to use to guide the prediction.
    public static final int INPUT_WINDOW_SIZE = 60;

    public static final int HIDDEN_LAYER_NEURONS = 10;

    /**
     * Normalize the mortgage rate values.
     */
    public static NormalizedField normRate = new NormalizedField(
            NormalizationAction.Normalize, "rate", 18.63, 2.65, HIGH, LOW);

    public static double[] getRates(File rawFile) {
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ',');

        double[] rates = new double[2693];
        int i = 0;
        while (csv.next()) {
            double interestRate = csv.getDouble(1);
            rates[i] = interestRate;

            i++;
        }
        csv.close();
        return rates;
    }

    public static int[] getDates(File rawFile) {
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ',');
        //2693
        int[] dates = new int[2693];
        int i = 0;
        while (csv.next()) {
            // Date format is yyyy-mm-dd, we'll pare it into yyyymmdd to be used as a sequence integer
            String dateString = csv.get(0);
            String modifiedDate = dateString.replace("-", "");
            int sequenceNumber = Integer.parseInt(modifiedDate);
            dates[i] = sequenceNumber;

            i++;
        }
        csv.close();
        return dates;
    }

    public MLDataSet generateTraining() {
        MLDataSet result = new BasicMLDataSet();

        for (int idx = TRAIN_START; idx < TRAIN_END; idx++) {
            if (idx < INPUT_WINDOW_SIZE) {
                continue;
            }
            MLData inputData = new BasicMLData(INPUT_WINDOW_SIZE);
            MLData idealData = new BasicMLData(1);

            // Look backward in training data to build up the input sliding window
            for (int j = 0; j < INPUT_WINDOW_SIZE; j++) {
                inputData.setData(j, normRate.normalize(RATES[idx - INPUT_WINDOW_SIZE + j]));
            }

            //inputData.setData(window_idx, normRate.normalize(RATES[idx]));
            //idealData.setData(0, normRate.normalize(RATES[idx + 1]));
            idealData.setData(0, normRate.normalize(RATES[idx]));

            result.add(inputData, idealData);
        }
        return result;
    }

    public BasicNetwork createNetwork() {
        ElmanPattern pattern = new ElmanPattern();
        pattern.setInputNeurons(INPUT_WINDOW_SIZE);
        pattern.addHiddenLayer(HIDDEN_LAYER_NEURONS);
        pattern.setOutputNeurons(1);
        pattern.setActivationFunction(new ActivationSigmoid());
        return (BasicNetwork)pattern.generate();
    }

    void train(BasicNetwork network, MLDataSet training) {
        final Train train = new ResilientPropagation(network, training);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #: " + epoch + "Error: " + train.getError());
            epoch++;
        } while(train.getError() > MAX_ERROR);

        System.out.println("Final Error: " + network.calculateError(training));

    }

    public void predict(BasicNetwork network) {
        double prediction = 0;
        System.out.printf("%11s, %9s, %8s, %8s, %8s\n", "train/test", "Year", "Actual", "Predict", "Error");

        BasicNetwork regular = (BasicNetwork)network.clone();
        BasicNetwork closedLoop = (BasicNetwork)network.clone();

        regular.clearContext();
        closedLoop.clearContext();

        for (int idx = 0; idx < RATES.length; idx++) {
            // We chose to start the for loop at idx = 0 rather than idx = INPUT_WINDOW_SIZE because it makes
            // the results lines up better for comparison with different window sizes
            if (idx >= INPUT_WINDOW_SIZE) {
                //Calculate based on actual data
                MLData input = new BasicMLData(INPUT_WINDOW_SIZE);
                for (int j = 0; j < INPUT_WINDOW_SIZE; j++) {
                    input.setData(j, normRate.normalize(RATES[idx - INPUT_WINDOW_SIZE + j]));
                }
                //input.setData(0, normRate.normalize(RATES[idx - 1]));

                MLData output = regular.compute(input);
                prediction = normRate.deNormalize(output.getData(0));
            }

            String t;

            if (idx < TESTING_START) {
                t = "Training";
            }
            else {
                t = "Testing";
            }

            double error = 0.5 * (prediction - RATES[idx]) * (prediction - RATES[idx]);

            //Display
            System.out.printf("%11s, %9d, %8.2f, %8.2f, %8.3f\n", t, DATES[idx], RATES[idx], prediction, error);
        }
    }

    public void run() {
        BasicNetwork network = createNetwork();
        MLDataSet training = generateTraining();
        train(network, training);
        predict(network);
    }

    public static void main(String args[]) {
        ElmanFRM frm = new ElmanFRM();
        frm.run();
    }
}
