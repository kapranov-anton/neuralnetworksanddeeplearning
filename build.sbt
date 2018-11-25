val breezeV = "0.13.2"

organization := "com.example"
scalaVersion := "2.12.7"
version := "0.1.0-SNAPSHOT"
name := "nn"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.5" % Test,
  "org.scalanlp" %% "breeze" % breezeV,
  "org.scalanlp" %% "breeze-natives" % breezeV,
  "org.scalanlp" %% "breeze-viz" % breezeV,
//  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
)
