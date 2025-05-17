import { Link, Stack } from 'expo-router';
import React from 'react';
import { Text, View,  StyleSheet } from 'react-native';

export default function NotFoundScreen() {
  return (
    <>
    <Stack.Screen options={{title: "Oops! Not Found.."}} />
    <View style={styles.container}>
        <Link href="/" style = {styles.button}>Navigate back to home page
        </Link>  
    </View>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#25292e',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: '#fff',
  },
  button: {
    fontSize: 20,
    textDecorationLine: "underline",
    color: "#fff",
  },
});
