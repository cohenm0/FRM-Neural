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
import org.encog.util.EngineArray;
import org.encog.util.arrayutil.NormalizeArray;
import org.encog.util.csv.ReadCSV;
import java.io.File;
import java.text.NumberFormat;
import java.util.Arrays;

public class ElmanFRM {

    public static final File MYDIR = new File("data/");

    public static File rawFile = new File(MYDIR, "MORTGAGE30US.csv");

    public final static double[] RATES = getRates(rawFile);

    public final static int STARTING_YEAR = 1991;

    public final static int TRAIN_START = 0;

    public final static int TRAIN_END = 2299;

    public final static int EVALUATE_START = 2300;

    public final static int EVALUATE_END = RATES.length-1;

    public final static double MAX_ERROR = 0.01;

    public final static double LOW = 0.1;

    public final static double HIGH = 0.9;

    private double[] normalizedRates;

    private double[] closedLoopRates;

    public static double[] getRates(File rawFile) {
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ',');
        double[] rates = new double[2693];
        int i = 0;
        while (csv.next()) {
            // Date format is yyyy-mm-dd, we'll pare it into yyyymmdd to be used as a sequence integer
            String dateString = csv.get(0);
            String modifiedDate = dateString.replace("-", "");
            int sequenceNumber = Integer.parseInt(modifiedDate);

            double interestRate = csv.getDouble(1);
            rates[i] = interestRate;
            i++;
        }
        csv.close();
        return rates;
    }

    public void normalizeRates(double low, double high) {
        NormalizeArray normalize = new NormalizeArray();
        normalize.setNormalizedHigh(high);
        normalize.setNormalizedLow(low);

        normalizedRates = normalize.process(RATES);
        closedLoopRates = EngineArray.arrayCopy(normalizedRates);
    }

    public MLDataSet generateTraining() {
        MLDataSet result = new BasicMLDataSet();

        for (int year = TRAIN_START; year < TRAIN_END; year++) {
            MLData inputData = new BasicMLData(1);
            MLData idealData = new BasicMLData(1);
            inputData.setData(0, this.normalizedRates[year]);
            idealData.setData(0, this.normalizedRates[year]);
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
    void train(BasicNetwork network,MLDataSet training) {
        final Train train = new ResilientPropagation(network, training);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + "Error" + train.getError());
            epoch++;
        } while(train.getError() > MAX_ERROR);

        System.out.println(network.calculateError(training));

    }
    public void predict(BasicNetwork network) {
        NumberFormat format = NumberFormat.getNumberInstance();
        format.setMaximumFractionDigits(4);
        format.setMinimumFractionDigits(4);

        System.out.println("Year\tActual\tPredict\tClosed Loop Predict");

        BasicNetwork regular = (BasicNetwork)network.clone();
        BasicNetwork closedLoop = (BasicNetwork)network.clone();

        regular.clearContext();
        closedLoop.clearContext();

        for (int year = 1; year < this.normalizedRates.length; year++) {
            //Calculate based on actual data
            MLData input = new BasicMLData(1);
            input.setData(0, this.normalizedRates[year - 1]);

            MLData output = regular.compute(input);
            double prediction = output.getData(0);
            this.closedLoopRates[year] = prediction;

            //Calculate the closed loop based on predicted data
            input.setData(0, this.closedLoopRates[year - 1]);
            output = closedLoop.compute(input);
            double closedLoopPrediction = output.getData(0);

            String t;

            if (year < EVALUATE_START) {
                t = "Train:";
            }
            else {
                t = "Evaluate:";
            }

            //Display
            System.out.println(t + (STARTING_YEAR + year) + "\t" + format.format(this.normalizedRates[year]) + "\t" + format.format(prediction) + "\t" + format.format(closedLoopPrediction));
        }
    }
    public void run() {
        normalizeRates(LOW, HIGH);
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
