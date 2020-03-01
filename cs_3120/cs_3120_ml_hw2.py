{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "from sklearn import model_selection\n",
    "from sklearn import linear_model\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "[[188  15]\n [ 51  54]]\n"
    }
   ],
   "source": [
    "cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'labelvalue']\n",
    "pima = pd.read_csv(\"./datasets/pima-indians-diabetes-database.csv\", header=None, names=cols)\n",
    "features = ['glucose', 'bp', 'insulin', 'bmi', 'age']\n",
    "\n",
    "X_train, X_test, Y_train, Y_test = model_selection.train_test_split(\n",
    "    pima[features].to_numpy(), pima.labelvalue, test_size=0.4\n",
    ")\n",
    "\n",
    "logreg = linear_model.LogisticRegression()\n",
    "logreg.fit(X_train, Y_train)\n",
    "\n",
    "Y_predict = logreg.predict(X_test)\n",
    "Y_proba = logreg.predict_proba(X_test)[::,1]\n",
    "fpr, tpr, _ = metrics.roc_curve(Y_test, Y_proba)\n",
    "\n",
    "print(metrics.confusion_matrix(Y_test, Y_predict))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "Accuracy:       0.7857142857142857\nPrecision:      0.782608695652174\nRecall:         0.5142857142857142\nHarmonic mean:  0.6206896551724138\n\n"
    }
   ],
   "source": [
    "accuracy = metrics.accuracy_score(Y_test, Y_predict)\n",
    "precision = metrics.precision_score(Y_test, Y_predict)\n",
    "recall = metrics.recall_score(Y_test, Y_predict)\n",
    "harmonic = 2*precision*recall/(precision + recall)\n",
    "\n",
    "print(f\"Accuracy:       {accuracy}\")\n",
    "print(f\"Precision:      {precision}\")\n",
    "print(f\"Recall:         {recall}\")\n",
    "print(f\"Harmonic mean:  {harmonic}\\n\")\n",
    "\n",
    "# print(metrics.classification_report(Y_test, Y_predict))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "type": "scatter",
         "x": [
          0,
          0,
          0,
          0.0049261083743842365,
          0.0049261083743842365,
          0.009852216748768473,
          0.009852216748768473,
          0.014778325123152709,
          0.014778325123152709,
          0.019704433497536946,
          0.019704433497536946,
          0.024630541871921183,
          0.024630541871921183,
          0.03940886699507389,
          0.03940886699507389,
          0.059113300492610835,
          0.059113300492610835,
          0.06403940886699508,
          0.06403940886699508,
          0.07389162561576355,
          0.07389162561576355,
          0.08374384236453201,
          0.08374384236453201,
          0.08866995073891626,
          0.08866995073891626,
          0.11330049261083744,
          0.11330049261083744,
          0.11822660098522167,
          0.11822660098522167,
          0.12315270935960591,
          0.12315270935960591,
          0.12807881773399016,
          0.12807881773399016,
          0.1330049261083744,
          0.1330049261083744,
          0.17733990147783252,
          0.17733990147783252,
          0.18226600985221675,
          0.18226600985221675,
          0.22660098522167488,
          0.22660098522167488,
          0.2315270935960591,
          0.2315270935960591,
          0.2561576354679803,
          0.2561576354679803,
          0.26108374384236455,
          0.26108374384236455,
          0.2955665024630542,
          0.2955665024630542,
          0.30049261083743845,
          0.30049261083743845,
          0.3054187192118227,
          0.3054187192118227,
          0.3103448275862069,
          0.3103448275862069,
          0.31527093596059114,
          0.31527093596059114,
          0.3399014778325123,
          0.3399014778325123,
          0.35960591133004927,
          0.35960591133004927,
          0.39901477832512317,
          0.39901477832512317,
          0.42857142857142855,
          0.42857142857142855,
          0.43842364532019706,
          0.43842364532019706,
          0.4630541871921182,
          0.4630541871921182,
          0.46798029556650245,
          0.46798029556650245,
          0.47783251231527096,
          0.47783251231527096,
          0.4827586206896552,
          0.4827586206896552,
          0.4975369458128079,
          0.4975369458128079,
          0.5812807881773399,
          0.5812807881773399,
          0.5911330049261084,
          0.5911330049261084,
          0.6157635467980296,
          0.6157635467980296,
          0.6354679802955665,
          0.6354679802955665,
          0.6748768472906403,
          0.6748768472906403,
          0.7931034482758621,
          0.7931034482758621,
          0.8078817733990148,
          0.8078817733990148,
          1
         ],
         "y": [
          0,
          0.009523809523809525,
          0.0380952380952381,
          0.0380952380952381,
          0.08571428571428572,
          0.08571428571428572,
          0.11428571428571428,
          0.11428571428571428,
          0.2,
          0.2,
          0.34285714285714286,
          0.34285714285714286,
          0.41904761904761906,
          0.41904761904761906,
          0.44761904761904764,
          0.44761904761904764,
          0.4857142857142857,
          0.4857142857142857,
          0.49523809523809526,
          0.49523809523809526,
          0.5238095238095238,
          0.5238095238095238,
          0.5333333333333333,
          0.5333333333333333,
          0.5428571428571428,
          0.5428571428571428,
          0.5523809523809524,
          0.5523809523809524,
          0.5619047619047619,
          0.5619047619047619,
          0.580952380952381,
          0.580952380952381,
          0.5904761904761905,
          0.5904761904761905,
          0.6,
          0.6,
          0.638095238095238,
          0.638095238095238,
          0.6476190476190476,
          0.6476190476190476,
          0.6571428571428571,
          0.6571428571428571,
          0.6666666666666666,
          0.6666666666666666,
          0.6857142857142857,
          0.6857142857142857,
          0.7047619047619048,
          0.7047619047619048,
          0.7142857142857143,
          0.7142857142857143,
          0.7238095238095238,
          0.7238095238095238,
          0.7333333333333333,
          0.7333333333333333,
          0.7428571428571429,
          0.7428571428571429,
          0.7523809523809524,
          0.7523809523809524,
          0.7619047619047619,
          0.7619047619047619,
          0.7714285714285715,
          0.7714285714285715,
          0.8095238095238095,
          0.8095238095238095,
          0.8285714285714286,
          0.8285714285714286,
          0.8380952380952381,
          0.8380952380952381,
          0.8571428571428571,
          0.8571428571428571,
          0.8666666666666667,
          0.8666666666666667,
          0.8761904761904762,
          0.8761904761904762,
          0.8952380952380953,
          0.8952380952380953,
          0.9047619047619048,
          0.9047619047619048,
          0.9238095238095239,
          0.9238095238095239,
          0.9333333333333333,
          0.9333333333333333,
          0.9523809523809523,
          0.9523809523809523,
          0.9619047619047619,
          0.9619047619047619,
          0.9714285714285714,
          0.9714285714285714,
          0.9809523809523809,
          0.9809523809523809,
          1,
          1
         ]
        }
       ],
       "layout": {
        "height": 950,
        "legend": {
         "bgcolor": "rgba(0,0,0,0.3)",
         "font": {
          "color": "white"
         },
         "x": 0,
         "y": 1
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "area under ROC curve: 0.8116819141449683"
        },
        "width": 950,
        "xaxis": {
         "title": {
          "text": "False Positive Rate"
         }
        },
        "yaxis": {
         "title": {
          "text": "True Positive Rate"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "roc = go.Scatter(x=fpr, y=tpr)\n",
    "auc = metrics.roc_auc_score(Y_test, Y_proba)\n",
    "fig = go.Figure(\n",
    "    data=roc,\n",
    "    layout=go.Layout(\n",
    "        width=950, height=950,\n",
    "        title=f\"area under ROC curve: {auc}\",\n",
    "        xaxis_title=\"False Positive Rate\",\n",
    "        yaxis_title=\"True Positive Rate\",\n",
    "        legend_x=0, legend_y=1,\n",
    "        legend_bgcolor='rgba(0,0,0,0.3)',\n",
    "        legend_font_color='white'\n",
    "    )\n",
    ")\n",
    "fig.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1-candidate"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}