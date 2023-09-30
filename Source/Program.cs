using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RunOnnxRuntimeConsole
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Get path to model to create inference session.
            var modelPath = args[0];

            // numOfDimensions
            int numOfDimensions = args.Count() - 1;

            // create input tensor (nlp example)
            double[] doubles = new double[numOfDimensions];

            for (int i = 0; i < numOfDimensions; i++)
            {
                doubles[i] = Convert.ToDouble(args[i+ 1]);
            }

            int[] dimensions = { numOfDimensions };
            var inputTensor = new DenseTensor<double>(doubles, dimensions);

            // Create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<double>("input", inputTensor) };

            // Create an InferenceSession from the Model Path.
            var session = new InferenceSession(modelPath);

            // Run session and send input data in to get inference output. Call ToList then get the Last item. Then use the AsEnumerable extension method to return the Value result as an Enumerable of NamedOnnxValue.
            //var output = session.Run(input).ToList().Last().AsEnumerable<NamedOnnxValue>();

            var result = session.Run(input);

            var r = ((DenseTensor<double>)result.Single().Value).ToArray();

            Console.WriteLine(r[0].ToString());
            //Console.WriteLine("Press 'Enter' To End");
            //Console.ReadLine();

            // From the Enumerable output create the inferenceResult by getting the First value and using the AsDictionary extension method of the NamedOnnxValue.
            //var inferenceResult = output.First().AsDictionary<string, double>();
        }
    }
}
