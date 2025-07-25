using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Windows.ApplicationModel;
using Windows.ApplicationModel.Activation;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace UniAli
{
    sealed partial class App : Application
    {
        public static InferenceSession Session { get; private set; }

        public App()
        {
            this.InitializeComponent();
            this.Suspending += OnSuspending;
            InitOnnxRuntime();
        }

        private async void InitOnnxRuntime()
        {
            try
            {
                var modelFile = await Package.Current.InstalledLocation.GetFileAsync(
                    @"Assets\model_q5.onnx");

                // Configure session options for better performance
                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = true
                };

                // Set execution provider - try CUDA first, fall back to CPU
                //try
                //{
                //    sessionOptions.AppendExecutionProvider_CUDA();
                //    Debug.WriteLine("CUDA execution provider initialized");
                //}
                //catch (Exception ex)
                //{
                //    Debug.WriteLine($"CUDA not available, falling back to CPU: {ex.Message}");
                    sessionOptions.AppendExecutionProvider_CPU();
                //}

                // Disable memory pattern for better compatibility
                sessionOptions.EnableMemoryPattern = false;

                // Load the model
                Session = new InferenceSession(modelFile.Path, sessionOptions);

                // Log model input/output info for debugging
                foreach (var input in Session.InputMetadata)
                {
                    //Debug.WriteLine($"Input: {input.Key}, Type: {input.Value.ElementType}, Dimensions: {string.Join(",", input.Value.Dimensions)}");
                }
            }
            catch (Exception ex)
            {
                // ��������� ������
                Debug.WriteLine("[ex] App exception: " + ex.Message);
            }
        }


        /// <summary>
        /// Invoked when the application is launched normally by the end user.  Other entry points
        /// will be used such as when the application is launched to open a specific file.
        /// </summary>
        /// <param name="e">Details about the launch request and process.</param>
        protected override void OnLaunched(LaunchActivatedEventArgs e)
        {
            Frame rootFrame = Window.Current.Content as Frame;

            // Do not repeat app initialization when the Window already has content,
            // just ensure that the window is active
            if (rootFrame == null)
            {
                // Create a Frame to act as the navigation context and navigate to the first page
                rootFrame = new Frame();

                rootFrame.NavigationFailed += OnNavigationFailed;

                if (e.PreviousExecutionState == ApplicationExecutionState.Terminated)
                {
                    //TODO: Load state from previously suspended application
                }

                // Place the frame in the current Window
                Window.Current.Content = rootFrame;
            }

            if (e.PrelaunchActivated == false)
            {
                if (rootFrame.Content == null)
                {
                    // When the navigation stack isn't restored navigate to the first page,
                    // configuring the new page by passing required information as a navigation
                    // parameter
                    rootFrame.Navigate(typeof(MainPage), e.Arguments);
                }
                // Ensure the current window is active
                Window.Current.Activate();
            }
        }

        /// <summary>
        /// Invoked when Navigation to a certain page fails
        /// </summary>
        /// <param name="sender">The Frame which failed navigation</param>
        /// <param name="e">Details about the navigation failure</param>
        void OnNavigationFailed(object sender, NavigationFailedEventArgs e)
        {
            throw new Exception("Failed to load Page " + e.SourcePageType.FullName);
        }

        /// <summary>
        /// Invoked when application execution is being suspended.  Application state is saved
        /// without knowing whether the application will be terminated or resumed with the contents
        /// of memory still intact.
        /// </summary>
        /// <param name="sender">The source of the suspend request.</param>
        /// <param name="e">Details about the suspend request.</param>
        private void OnSuspending(object sender, SuspendingEventArgs e)
        {
            var deferral = e.SuspendingOperation.GetDeferral();
            //TODO: Save application state and stop any background activity
            deferral.Complete();
        }
    }
}
