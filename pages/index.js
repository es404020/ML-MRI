import { Inter } from "next/font/google";
import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
const inter = Inter({ subsets: ["latin"] });

export default function Home() {
  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState("No Tumor");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const categories = ["Bengin", "Malignant", "No Tumor"];
  const [image, setImage] = useState("/Normal case (213).jpg");

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    setLoading(true);
    try {
      const model = await tf.loadLayersModel(
        "https://mri-lung-detection.s3.us-east.cloud-object-storage.appdomain.cloud/model.json"
      );
      setModel(model);
      model.summary();
      setLoading(false);
    } catch (err) {
      setLoading(false);
      setError("error");
    }
  };

  const hiddenFileInput = useRef(null);

  const readImageFile = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();

      reader.onload = () => resolve(reader.result);

      reader.readAsDataURL(file);
    });
  };
  const createHTMLImageElement = (imageSrc) => {
    return new Promise((resolve) => {
      const img = new Image();

      img.onload = () => resolve(img);

      img.src = imageSrc;
    });
  };

  const handleChange = async (event) => {
    if (event.target.files && event.target.files[0]) {
      const i = event.target.files[0];
      var reader = new FileReader();
      var url = reader.readAsDataURL(i);
      reader.onloadend = function (e) {
        setImage(reader.result);
      };

      const imageSrc = await readImageFile(i);
      const image = await createHTMLImageElement(imageSrc);
      await tf.tidy(() => {
        // let img = tf.browser.fromPixels(image, 1);
        // img = img.reshape([1, 256, 256, 3]);
        // img = tf.cast(img, "float32");

        const tensorImg = tf.browser
          .fromPixels(image, 1)
          .resizeNearestNeighbor([256, 256])
          .toFloat()
          .expandDims();
        const result = model.predict(tensorImg);

        // const outputs = model.predict(img);
        const pros = tf.softmax(Array.from(result.dataSync()));
        const arr = Array.from(result.dataSync());

        const max = Math.max(...arr);

        const index = arr.indexOf(max);
        setClassLabels(categories[index]);
        pros.print();
      });
    }
  };

  const handleClick = (event) => {
    
    hiddenFileInput.current.click();
  };

  return (
    <>
      {error ? (
         <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>{ error }</p>
      ) : !loading ? (
        <>
          <main
            className={`flex min-h-screen flex-col items-center justify-between p-24 ${inter.className}`}
          >
            <input
              type="file"
              ref={hiddenFileInput}
              onChange={handleChange}
              accept="image/png, image/jpeg"
              style={{ display: "none" }}
            />
            <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
              <p
                onClick={handleClick}
                className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30"
              >
                Click to upload &nbsp;
                <code className="font-mono font-bold">MRI image </code>
              </p>
              <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
                <a
                  className="pointer-events-none flex place-items-center gap-2 p-8 lg:pointer-events-auto lg:p-0"
                  href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template-tw&utm_campaign=create-next-app"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Result: {classLabels && <>{classLabels}</>}
                  {/* <Image
                src='/vercel.svg'
                alt="Vercel Logo"
                className="dark:invert"
                width={100}
                height={24}
                priority
              /> */}
                </a>
              </div>
            </div>

            <div className="relative flex place-items-center before:absolute before:h-[300px] before:w-[480px] before:-translate-x-1/2 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700/10 after:dark:from-sky-900 after:dark:via-[#0141ff]/40 before:lg:h-[360px]">
              <img
                className="relative  mb-4"
                src={image}
                alt="Next.js Logo"
                width={280}
                height={280}
                priority
              />
            </div>

            <div className="mb-32 grid text-center lg:mb-0 lg:grid-cols-4 lg:text-left">
              <a
                href="hhttps://colab.research.google.com/drive/15ghF1kDZARoyEXaRFzJBXMPemGF39JzJ?usp=sharing"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                target="_blank"
                rel="noopener noreferrer"
              >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                  Docs{" "}
                  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                  </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                  Explore this jupiter notebook to undestand how this model was
                  built.
                </p>
              </a>

              <a
                href="https://www.tensorflow.org/"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                target="_blank"
                rel="noopener noreferrer"
              >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                  Learn{" "}
                  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                  </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                  Learn about tensorflow and tensorflow.js required to build
                  deep learning models.
                </p>
              </a>

              <a
                href="https://www.youtube.com/watch?v=tUHl5TiP_oA&ab_channel=NicholasRenotte"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                target="_blank"
                rel="noopener noreferrer"
              >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                  Deployment{" "}
                  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                  </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                  Discover and deploy your ML models&nbsp;.
                </p>
              </a>

              <a
                download
                href="/Malignant case (1).jpg"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                target="_blank"
                rel="noopener noreferrer"
              >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                  Test Images{" "}
                  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                  </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                  Download random MRI images for test.
                </p>
              </a>
            </div>
          </main>
        </>
      ) : (
        <>
          <main
            className={`flex min-h-screen flex-col items-center justify-between p-24 ${inter.className}`}
          >
            <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
              Fetching model ......
            </p>
          </main>
        </>
      )}
    </>
  );
}
