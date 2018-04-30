from bokeh.embed import components
from bokeh.models import FuncTickFormatter, FixedTicker, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

#Hex codes for primary seaborn color palette (I personally like them)
COLORS = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD',
          '#8C564B','#E377C2','#7F7F7F','#BCBD22','#17BECF']

def trend_plot(data, width=600, height=300):
    '''
    '''
    plot = figure(width=width, height=height, tools=['save'])
    if len(data) > 5:
        data = data[:5]
    for i, d in enumerate(data):
        plot.line(d['years'], d['counts'], legend='Topic {}'.format(d['id']),
                  line_color=COLORS[i], line_width=2.)
    plot.title.text = '# of grants in similar topics by year'
    plot.legend.location = 'top_left'
    plot.grid.grid_line_alpha = 0
    plot.xaxis.axis_label = 'Year'
    plot.yaxis.axis_label = '# of grants'
    plot.xaxis.axis_label_text_font_size = "1.5em"
    plot.yaxis.axis_label_text_font_size = "1.5em"
    plot.xaxis.major_label_text_font_size = "1em"
    return components(plot)


def histogram_plot(data, width=600, height=300):
    '''
    '''
    if len(data) > 5:
        data = data[:5]
    ids = [d['id'] for d in data]
    topics = ['Topic {}'.format(d['id']) for d in data]
    dist = [d['dist'] for d in data]
    labels = [', '.join(d['words']) for d in data]
    source = ColumnDataSource(data=dict(topics=topics, distribution=dist,
                                        ids=ids, labels=labels))
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <span style="font-size: 20px;">Topics: <strong>@ids</strong></span><br>
                <span style="font-size: 20px;">Words: @labels</span>
            </div>
        </div>
    """)
    hover.attachment = 'right'

    plot = figure(x_range=topics, width=width, height=height, tools=[hover, 'save'],
                  title='Relevant topics')
    plot.vbar(x='topics', top='distribution', width=0.8, source=source, line_color='white',
              fill_color=factor_cmap('topics', palette=COLORS, factors=topics))
    plot.y_range.start = 0
    plot.xgrid.grid_line_color = None
    plot.xaxis.axis_label = 'Topics'
    plot.yaxis.axis_label = 'Portion'
    plot.xaxis.axis_label_text_font_size = "1.5em"
    plot.yaxis.axis_label_text_font_size = "1.5em"
    plot.xaxis.major_label_text_font_size = "1em"
    plot.toolbar.active_inspect = [hover]
    return components(plot)
