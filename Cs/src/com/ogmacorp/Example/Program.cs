// -----------------------------------------------------------------------------
// OgmaNeo
// Copyright (c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
// -----------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ogmaneo;

namespace Example
{
    class Program
    {
        static void Main(string[] args)
        {
            int numSimSteps = 500;
            ComputeSystem.DeviceType deviceType = ComputeSystem.DeviceType._gpu;

            Resources _res = new Resources();
            _res.create(deviceType);

            Architect arch = new Architect();
            arch.initialize(1234, _res);

            // Input size (width and height)
            int w = 4;
            int h = 4;

            ParameterModifier inputParams = arch.addInputLayer(new Vec2i(w, h));
            inputParams.setValue("in_p_alpha", 0.02f);
            inputParams.setValue("in_p_radius", 16);

            for (int i = 0; i < 2; i++)
            {
                ParameterModifier layerParams = arch.addHigherLayer(new Vec2i(36, 36), SparseFeaturesType._chunk);
                layerParams.setValue("sfc_chunkSize", new Vec2i(6, 6));
                layerParams.setValue("sfc_ff_radius", 12);
                layerParams.setValue("hl_poolSteps", 2);
                layerParams.setValue("sfc_weightAlpha", 0.02f);
                layerParams.setValue("sfc_biasAlpha", 0.001f);
                layerParams.setValue("p_alpha", 0.08f);
                layerParams.setValue("p_beta", 0.16f);
                layerParams.setValue("p_radius", 16);
            }

            Hierarchy hierarchy = arch.generateHierarchy();

            ValueField2D inputField = new ValueField2D(new Vec2i(w, h));

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    inputField.setValue(new Vec2i(x, y), (y * w) + x);
                }
            }

            vectorvf inputVector = new vectorvf();
            inputVector.Add(inputField);

            System.Console.WriteLine("Stepping the hierarchy...");
            for (int i = 0; i < numSimSteps; i++)
            {
                hierarchy.simStep(inputVector, true);
            }

            vectorvf prediction = hierarchy.getPredictions();

            System.Console.Write("Input      :");
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    System.Console.Write(" " + inputField.getValue(new Vec2i(x, y)).ToString("n2"));
                }
            }
            System.Console.WriteLine();

            System.Console.Write("Prediction :");
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    System.Console.Write(" " + prediction[0].getValue(new Vec2i(x, y)).ToString("n2"));
                }
            }
            System.Console.WriteLine();
        }
    }
}
