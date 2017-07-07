// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

import java.io.File;
import com.ogmacorp.ogmaneo.*;

public class Example {

    static private boolean serializationEnabled = false;

    public static void main(String[] args) {
        int numSimSteps = 100;

        ComputeSystem.DeviceType deviceType = ComputeSystem.DeviceType._gpu;

        if (args.length > 0) {
            try {
                numSimSteps = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be an integer.");
                System.exit(1);
            }

            // Travis-CI passes 0 as an arg to allow for testing of the bindings
            if (numSimSteps == 0)
                deviceType = ComputeSystem.DeviceType._cpu;
        }

        Resources _res = new Resources();
        _res.create(deviceType);

        Architect arch = new Architect();
        arch.initialize(1234, _res);

        // Input size (width and height)
        int w = 4;
        int h = 4;

        ParameterModifier inputParams = arch.addInputLayer(new Vec2i(w, h));
        inputParams.setValue("in_p_alpha", 0.02f);
        inputParams.setValue("in_p_radius", 8);

        for (int i = 0; i < 2; i++) {
            ParameterModifier layerParams = arch.addHigherLayer(new Vec2i(32, 32), SparseFeaturesType._chunk);
            layerParams.setValue("sfc_numSamples", 2);
        }

        Hierarchy hierarchy = arch.generateHierarchy();

        ValueField2D inputField = new ValueField2D(new Vec2i(w, h));

        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                inputField.setValue(new Vec2i(x, y), (y * w) + x);
            }
        }

        if (serializationEnabled) {
            File f = new File("Example.opr");
            if (f.exists() && !f.isDirectory()) {
                System.out.println("Loading hierarchy from Example.opr");
                hierarchy.load(_res.getComputeSystem(), "Example.opr");
            }
        }

        System.out.println("Stepping the hierarchy...");
        for (int i = 0; i < numSimSteps; i++) {
            vectorvf inputVector = new vectorvf();
            inputVector.add(inputField);

            hierarchy.activate(inputVector);
            hierarchy.learn(inputVector);
            //System.out.print(".");
        }
        System.out.println();

        ValueField2D prediction = hierarchy.getPredictions().get(0);

        if (numSimSteps > 0) {
            System.out.print("Input      :");
            for(int y = 0; y < h; y++) {
                for(int x = 0; x < w; x++) {
                    System.out.printf(" %.2f", inputField.getValue(new Vec2i(x, y)));
                }
            }
            System.out.println();

            System.out.print("Prediction :");
            for(int y = 0; y < h; y++) {
                for(int x = 0; x < w; x++) {
                    System.out.printf(" %.2f", prediction.getValue(new Vec2i(x,y)));
                }
            }
            System.out.println();

            if (serializationEnabled) {
                System.out.println("Saving hierarchy to Example.opr");
                hierarchy.save(_res.getComputeSystem(), "Example.opr");
            }
        }
    }
}
