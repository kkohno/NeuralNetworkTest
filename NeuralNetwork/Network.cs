namespace NeuralNetwork;

public class Network
{
    // Функция активации (сигмоида) и её производная
    static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    static double SigmoidDerivative(double x) => x * (1 - x);
    //static double Sigmoid(double x) => Math.Tanh(x);
    //static double SigmoidDerivative(double x) => 1 / (Math.Cosh(x) * Math.Cosh(x));

    public void Work()
    {
        // Данные: входы (X) и истинные значения (y)
        double[,] X = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }; // Логическая операция AND
        double[] y = { 0, 0, 0, 1 }; // Результат AND

        // Параметры сети
        int inputDim = 2; // Число входов
        int hiddenDim = 2; // Число нейронов в скрытом слое
        int outputDim = 1; // Число выходов
        double learningRate = 0.1;

        // Инициализация весов и смещений (случайно)
        Random rand = new Random();
        double[,] weightsInputHidden = new double[inputDim, hiddenDim];
        double[] weightsHiddenOutput = new double[hiddenDim];
        double[] biasHidden = new double[hiddenDim];
        double biasOutput = rand.NextDouble();

        // Заполнение весов случайными значениями
        for (int i = 0; i < inputDim; i++)
            for (int j = 0; j < hiddenDim; j++)
                weightsInputHidden[i, j] = rand.NextDouble();

        for (int i = 0; i < hiddenDim; i++)
            weightsHiddenOutput[i] = rand.NextDouble();

        for (int i = 0; i < hiddenDim; i++)
            biasHidden[i] = rand.NextDouble();

        // Обучение
        int epochs = 10000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;

            for (int sample = 0; sample < X.GetLength(0); sample++) {
                // Прямое распространение
                double[] hiddenInput = new double[hiddenDim];
                double[] hiddenOutput = new double[hiddenDim];

                for (int i = 0; i < hiddenDim; i++) {
                    hiddenInput[i] = biasHidden[i];
                    for (int j = 0; j < inputDim; j++)
                        hiddenInput[i] += X[sample, j] * weightsInputHidden[j, i];
                    hiddenOutput[i] = Sigmoid(hiddenInput[i]);
                }

                double finalInput = biasOutput;
                for (int i = 0; i < hiddenDim; i++)
                    finalInput += hiddenOutput[i] * weightsHiddenOutput[i];
                double finalOutput = Sigmoid(finalInput);

                // Ошибка
                double error = y[sample] - finalOutput;
                totalError += error * error;

                // Обратное распространение
                double outputGradient = error * SigmoidDerivative(finalOutput);
                double[] hiddenGradient = new double[hiddenDim];

                for (int i = 0; i < hiddenDim; i++)
                    hiddenGradient[i] = outputGradient * weightsHiddenOutput[i] * SigmoidDerivative(hiddenOutput[i]);

                // Обновление весов
                for (int i = 0; i < hiddenDim; i++)
                    weightsHiddenOutput[i] += learningRate * outputGradient * hiddenOutput[i];
                biasOutput += learningRate * outputGradient;

                for (int i = 0; i < hiddenDim; i++) {
                    for (int j = 0; j < inputDim; j++)
                        weightsInputHidden[j, i] += learningRate * hiddenGradient[i] * X[sample, j];
                    biasHidden[i] += learningRate * hiddenGradient[i];
                }
            }

            // Вывод ошибки каждые 1000 эпох
            if (epoch % 1000 == 0)
                Console.WriteLine($"Эпоха {epoch}, ошибка: {totalError / X.GetLength(0):F5}");
        }

        // Тестирование
        Console.WriteLine("\nТестирование сети:");
        for (int sample = 0; sample < X.GetLength(0); sample++) {
            double[] hiddenInput = new double[hiddenDim];
            double[] hiddenOutput = new double[hiddenDim];

            for (int i = 0; i < hiddenDim; i++) {
                hiddenInput[i] = biasHidden[i];
                for (int j = 0; j < inputDim; j++)
                    hiddenInput[i] += X[sample, j] * weightsInputHidden[j, i];
                hiddenOutput[i] = Sigmoid(hiddenInput[i]);
            }

            double finalInput = biasOutput;
            for (int i = 0; i < hiddenDim; i++)
                finalInput += hiddenOutput[i] * weightsHiddenOutput[i];
            double finalOutput = Sigmoid(finalInput);

            Console.WriteLine($"Входы: [{X[sample, 0]}, {X[sample, 1]}], Предсказание: {finalOutput:F5}, Истина: {y[sample]}");
        }
    }
}