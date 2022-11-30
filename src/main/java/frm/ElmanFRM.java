package frm;

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
            MLData inputData = new BasicMLData(1);
            MLData idealData = new BasicMLData(1);
            inputData.setData(0, normRate.normalize(RATES[idx]));
            idealData.setData(0, normRate.normalize(RATES[idx + 1]));
            result.add(inputData, idealData);
        }
        return result;
    }

    public BasicNetwork createNetwork() {
        ElmanPattern pattern = new ElmanPattern();
        pattern.setInputNeurons(1);
        pattern.addHiddenLayer(10);
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
        System.out.printf("%11s, %9s, %8s, %8s\n", "train/test", "Year", "Actual", "Predict");

        BasicNetwork regular = (BasicNetwork)network.clone();
        BasicNetwork closedLoop = (BasicNetwork)network.clone();

        regular.clearContext();
        closedLoop.clearContext();

        for (int idx = 1; idx < RATES.length; idx++) {
            //Calculate based on actual data
            MLData input = new BasicMLData(1);
            input.setData(0, normRate.normalize(RATES[idx - 1]));

            MLData output = regular.compute(input);
            double prediction = normRate.deNormalize(output.getData(0));

            String t;

            if (idx < TESTING_START) {
                t = "Training";
            }
            else {
                t = "Testing";
            }

            //Display
            System.out.printf("%11s, %9d, %8.2f, %8.2f\n", t, DATES[idx], RATES[idx], prediction);
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
