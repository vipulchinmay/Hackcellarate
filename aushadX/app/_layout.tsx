import { Stack } from "expo-router";
import React, { useState } from "react";
import { LogBox, View } from "react-native";
import { LanguageProvider } from "@/components/LanguageContext";
import SplashScreenComponent from "@/components/SplashScreen"; // Import splash screen
import 'react-native-gesture-handler';
// Rest of your imports

LogBox.ignoreAllLogs(true);

export default function RootLayout() {
  const [showSplash, setShowSplash] = useState(true);

  return showSplash ? (
    <SplashScreenComponent onFinish={() => setShowSplash(false)} />
  ) : (
    <LanguageProvider>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="+not-found" options={{ headerShown: false }} />
      </Stack>
    </LanguageProvider>
  );
}
